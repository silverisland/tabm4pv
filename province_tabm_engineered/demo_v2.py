import copy
import datetime
import logging
import os
import sys
import time
from typing import Optional

import yaml
import warnings

import pandas as pd
import numpy as np

sys.path.append("/home/nfdw/nuoya/algo/nuoya_algo/ensemble_model")
sys.path.append("/home/nfdw/nuoya/")

from weather_aggregation_rules_v2_1 import RULES
from feature_aggregation import aggregate_features

from detect_data_abnormal import check_ds_format_abnormal
from ods_transform.dataset_inferface import DataSetInterface
from inference import multi_station_inference


from api import predict as ultra_province_predict

warnings.filterwarnings("ignore")
multi_logger = logging.getLogger('PredictTool.model1')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

INFO_PATH = r"/jtdata/products/data/info.csv"


class MultiInfer:
    def __init__(
            self,
            report_time: str,
            data_path: dict = None,
            predict_type: str = 'short',
            checkpoints_dir: str = "",
            config: dict | str = None,
            train_data_saved_dir: str = "",
            infer_data_save_dir: str = "",
            save_path: str = "",
            timeout: Optional[int] = None,
            log_path: str = "",
    ):
        # 日志
        log_path = "/home/nfdw/nuoya/logs/"
        log_filename = f"multi_infer_{predict_type}_{report_time.replace(' ', '_').replace(':', '')}.log"
        # checkpoint
        self.checkpoints_dir = "/home/nfdw/nuoya/algo/checkpoints/"
        self.province_checkpoints_dir = "/home/nfdw/nuoya/algo/checkpoints/ckpt_province"
        # ods和ds保存路径
        self.train_data_saved_dir = "/home/nuoya_v1/data1/train"
        self.infer_data_save_dir = "/home/nuoya_v1/data1/infer"
        # 结果保存
        save_path = "/jtdata/products/data/result/nuoya"
        # 输入气象源
        self.data_path = {
            "power": "",
            "observe_weather": "",
            "SOLARGIS": "",
            "solar_southern": "",
            "GDFS_JTHFGNED_MID_JTNWQISX": "",
            "EC_JTKRD7RO": "",
        }
        # 需要预测的区域
        self.areas = ["province_guangxi_solar"]
        # nuoya配置文件
        self.config = "/home/nfdw/nuoya/algo/nuoya_algo/ensemble_model/chunk_config.yaml"

        self.report_time = report_time
        self.logger = multi_logger
        os.makedirs(log_path, exist_ok=True)
        log_file = os.path.join(log_path, log_filename)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        self.logger.info(f"日志文件：{log_file}")

        self.save_path = os.path.join(save_path, predict_type)

        # 历史/预报长度
        self.history_len = 672
        self.predict_type = predict_type
        if self.predict_type == 'short':
            self.predict_seq_lens = 480
            self.logger.info(f"设置预测长度：{self.predict_seq_lens}")

        elif self.predict_type == 'ultra_short':
            self.predict_seq_lens = 192
            self.logger.info(f"设置预测长度：{self.predict_seq_lens}")
        else:
            self.predict_seq_lens = 480

        self.logger.info(f"预测类型：{self.predict_type}， 设置预测长度：{self.predict_seq_lens}，"
                         f"使用气象源：{self.data_path.keys().__str__()}")

        # 起报时间
        self.forecast_time = datetime.datetime.strptime(self.report_time, "%Y-%m-%d %H:%M:%S")
        # 起报时间下一个时间点上开始预测的第一个点
        self.forecast_s_time = self.forecast_time + datetime.timedelta(minutes=15)

        # 构造预测结果保持点时间段
        if predict_type == 'short':
            self.result_s_time = (self.forecast_time + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0)
            self.result_e_time = (self.forecast_time + datetime.timedelta(days=4)).replace(hour=23, minute=45, second=0)
        else:
            self.result_s_time = self.forecast_s_time
            self.result_e_time = self.forecast_s_time + datetime.timedelta(hours=5) - datetime.timedelta(minutes=15)

        with open(self.config, 'r', encoding="utf-8") as f:
            self.config_dict = yaml.safe_load(f)

        self.only_power = False

    def weighted_weather(
            self,
            df: pd.DataFrame,
            area: str,
            station_prefix: str,
            value_cols: list[str],
    ) -> pd.DataFrame:
        """按真实场站装机容量聚合天气，并写回省级行。"""
        province = df[df["station"].eq(area)]
        if len(province) != 1:
            self.logger.warning(f"station={area} 的省级行数量为 {len(province)}，跳过天气加权")
            return df

        source = df[df["station"].str.startswith(station_prefix, na=False)]
        if source.empty:
            self.logger.warning(f"station_prefix={station_prefix} 未找到场站行，跳过天气加权")
            return df

        weighted_values = {}
        for timestamp_win, group in source.groupby("timestamp_win", sort=False):
            try:
                weights = group["cap_power_on"].map(
                    lambda value: np.asarray(value, dtype=float).reshape(-1)[0]
                ).to_numpy(dtype=float)
                invalid_mask = ~np.isfinite(weights)
                if invalid_mask.all():
                    raise ValueError("所有场站的 cap_power_on 均无效")
                if invalid_mask.any():
                    invalid_stations = group.loc[invalid_mask, "station"].tolist()
                    weights[invalid_mask] = weights[~invalid_mask].mean()
                    self.logger.info(f"容量异常站点：{invalid_stations}")

                weighted_values[timestamp_win] = {
                    col: np.ma.average(
                        np.ma.masked_invalid(np.stack(group[col].to_numpy())),
                        axis=0,
                        weights=weights,
                    ).filled(np.nan)
                    for col in value_cols
                }
            except Exception as error:
                self.logger.warning(
                    f"timestamp_win={timestamp_win} 天气加权失败，使用省级原值：{error}"
                )
                weighted_values[timestamp_win] = {
                    col: province.iloc[0][col] for col in value_cols
                }

        province_mask = df["station"].eq(area)
        for col in value_cols:
            df.loc[province_mask, col] = pd.Series(
                [
                    weighted_values[timestamp_win][col]
                    for timestamp_win in df.loc[province_mask, "timestamp_win"]
                ],
                index=df.index[province_mask],
                dtype=object,
            )
        return df

    def merge_province_predictions(
            self,
            area: str,
            tabm_result: pd.DataFrame | None,
            moe_result: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """保留完整MoE结果，仅用TabM有效值覆盖省级预测列。"""
        prediction_col = f"predict_power_{area}"
        result = moe_result.copy()
        tabm_mask = pd.Series(False, index=result.index)
        if tabm_result is not None and not tabm_result.empty:
            tabm_values = result["dtime"].map(
                tabm_result.set_index("dtime")[prediction_col]
            )
            tabm_values = pd.to_numeric(tabm_values, errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            tabm_mask = tabm_values.notna()
            result.loc[tabm_mask, prediction_col] = tabm_values[tabm_mask]

        first_20_mask = tabm_mask.iloc[:20].to_numpy()
        tabm_points = pd.DataFrame({
            "point": np.flatnonzero(first_20_mask) + 1,
            "dtime": result["dtime"].iloc[:20].to_numpy()[first_20_mask],
        })
        print("前20个点中使用TabM的点：")
        print(tabm_points.to_string(index=False) if len(tabm_points) else "无")
        return result

    def handle(self):
        start_t = time.time()
        self.logger.info(
            f"multi_infer: start to handle..., params: "
            f"report_time: {self.report_time}, "
            f"data_path: {self.data_path}, "
            f"save_path: {self.save_path}, "
            f"predict_type: {self.predict_type}, "
            f"predict_seq_lens: {self.predict_seq_lens}"
            f"checkpoints_dir: {self.checkpoints_dir}, "
            f"config: {self.config},"
            f"only_power: {self.only_power}"
        )

        stations_df = pd.read_csv(INFO_PATH)
        stations_df = stations_df[["plant_pointname", "LONGITUDE", "LATITUDE", "GCCAPACITY", "province_pointname"]]
        stations_df.rename(
            columns={'plant_pointname': 'plantId', 'LONGITUDE': 'lng', 'LATITUDE': 'lat', 'GCCAPACITY': 'cap_power_on'},
            inplace=True)

        error_area = []  # 统计预测异常的区域
        for area in self.areas:
            # 区分solar和wind
            area_type = "_solar" if area.endswith("_solar") else "_wind"
            province = area.replace(area_type, "")
            station_name_suffix = "plant_guangfu" if area.endswith("_solar") else "plant_fengdian"
            station_df = stations_df[stations_df["province_pointname"] == province]
            station_df = station_df[station_df["plantId"].str.startswith(station_name_suffix)]
            area_df = station_df[["plantId", "lng", "lat", "cap_power_on"]].copy()

            self.logger.info(f"=======start predict area: {area}, area_df shape: {area_df.shape}=======")
            df, success_flag = DataSetInterface(
                save_history_data_dir=self.train_data_saved_dir,
                save_infer_data_dir=self.infer_data_save_dir,
                logger=self.logger,
            ).prepare_infer_data(
                forecast_time=self.report_time,
                predict_type=self.predict_type,
                data_path=self.data_path,
                area_name=area,
                plants_df=area_df,
                predict_seq_lens=self.predict_seq_lens,
                history_len=self.history_len,
                only_power=self.only_power,
            )

            if not success_flag:
                self.logger.info(
                    f"multi_infer: finish generate infer area failed, area name: {area}, cost time: {time.time() - start_t}")
                error_area.append(area)
                continue

            self.logger.info(
                f"multi_infer: finish generate infer data..., shape: {df.shape}, cost time: {time.time() - start_t}")

            self.logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>RULES')
            self.logger.info(RULES)
            df = aggregate_features(df, RULES)
            for i in range(9):
                df[f'ssrd_pos_{i + 1}'] = df['solar_southern_GHI']
                df[f't2m_pos_{i + 1}'] = df['solar_southern_TEMP']
                df[f'ssrd_pos_{i + 1}_predict'] = df['solar_southern_GHI_predict']
                df[f't2m_pos_{i + 1}_predict'] = df['solar_southern_TEMP_predict']

            df['GHI_SOLARGIS'] = df['general_GHI']
            df['TEMP_SOLARGIS'] = df['general_TEMP']
            df['GHI_SOLARGIS_predict'] = df['general_GHI_predict']
            df['TEMP_SOLARGIS_predict'] = df['general_TEMP_predict']
            self.logger.info(df.iloc[0])

            try:  # 保存DS数据
                save_cols = [
                    'timestamp_win',
                    'station',
                    'observe_power',
                    'observe_power_future',
                    'ssrd_pos_1',
                    'ssrd_pos_2',
                    'ssrd_pos_3',
                    'ssrd_pos_4',
                    'ssrd_pos_5',
                    'ssrd_pos_6',
                    'ssrd_pos_7',
                    'ssrd_pos_8',
                    'ssrd_pos_9',
                    't2m_pos_1',
                    't2m_pos_2',
                    't2m_pos_3',
                    't2m_pos_4',
                    't2m_pos_5',
                    't2m_pos_6',
                    't2m_pos_7',
                    't2m_pos_8',
                    't2m_pos_9',
                    'GHI_SOLARGIS',
                    'TEMP_SOLARGIS',
                    'ssrd_pos_1_predict',
                    'ssrd_pos_2_predict',
                    'ssrd_pos_3_predict',
                    'ssrd_pos_4_predict',
                    'ssrd_pos_5_predict',
                    'ssrd_pos_6_predict',
                    'ssrd_pos_7_predict',
                    'ssrd_pos_8_predict',
                    'ssrd_pos_9_predict',
                    't2m_pos_1_predict',
                    't2m_pos_2_predict',
                    't2m_pos_3_predict',
                    't2m_pos_4_predict',
                    't2m_pos_5_predict',
                    't2m_pos_6_predict',
                    't2m_pos_7_predict',
                    't2m_pos_8_predict',
                    't2m_pos_9_predict',
                    'GHI_SOLARGIS_predict',
                    'TEMP_SOLARGIS_predict',
                ]
                save_ds_data_cache(
                    df=df[save_cols], report_time=self.report_time, area=area, predict_type=self.predict_type,
                    base_save_path=self.train_data_saved_dir
                )
            except Exception as e:
                self.logger.error(f"multi_infer: save ds data failed, error msg: {e}")


            # 保存加权前的省级数据，供旧省级模型作为第二行输入。
            unknown_province = df.loc[df["station"].eq(area)].copy()
            unknown_province.loc[:, "station"] = "unknown"

            # unknown 不参与天气加权；加权完成后再放回 df 一起检查。
            df = df.loc[~df["station"].eq("unknown")].copy()
            value_cols = [f"ssrd_pos_{i+1}_predict" for i in range(0, 9)] + ["GHI_SOLARGIS_predict"]
            df = self.weighted_weather(df, area, station_name_suffix, value_cols)
            df = pd.concat([df, unknown_province.iloc[[0]]], ignore_index=True)

            # 先检查数据，然后把站点分2类，异常类走tsfm，正常类走MoE V3.1
            if self.only_power:
                df_abnormal_part = df.copy()
                df_normal_part = pd.DataFrame()
                self.logger.info(f"multi_infer: only_power模式，跳过数据检查，全部数据送入tsfm: {df_abnormal_part.shape}")
            else:
                self.logger.info(f"multi_infer: 数据检查开始： {time.time() - start_t} s")
                self.logger.info(f"multi_infer: 数据检查开始 shape: {df.shape}")

                col_his, col_pred = [], []
                for col in df.columns:
                    if col in ['timestamp_win', 'station', 'observe_power_future']:
                        continue
                    if col.endswith('_predict'):
                        col_pred.append(col)
                    else:
                        col_his.append(col)

                # 传入检查长度
                df_result = check_ds_format_abnormal(df[['timestamp_win', 'station'] + col_his + col_pred],
                                                     pred_col_len=self.predict_seq_lens)
                province_check_mask = df_result['station'].isin([area, 'unknown'])
                if df_result.loc[province_check_mask, 'is_abnormal'].any():
                    df_result.loc[province_check_mask, 'is_abnormal'] = True
                    self.logger.warning("province/unknown 任一行异常，两行统一进入异常分支")
                self.logger.info(f"multi_infer: 数据检查结束： {time.time() - start_t} s")

                # 异常数据
                df_abnormal = df_result[df_result['is_abnormal'] == True]
                df_abnormal_part = df.merge(df_abnormal, on=['timestamp_win', 'station'], how='inner').drop(
                    columns=['is_abnormal'])
                self.logger.info(
                    f"发现异常数据 shape: {df_abnormal_part.shape} ， 站点：{df_abnormal_part['station'].tolist()}")

                # 正常数据
                df_normal = df_result[df_result['is_abnormal'] == False]
                df_normal_part = df.merge(df_normal, on=['timestamp_win', 'station'], how='inner').drop(
                    columns=['is_abnormal'])
                self.logger.info(f"正常数据 shape: {df_normal_part.shape} ")

            # 提取检查正常的 province 和 unknown，二者只进入省级模型。
            df_province = pd.DataFrame()
            if not df_normal_part.empty:
                province_mask = df_normal_part['station'].isin([area, 'unknown'])
                province_rows = df_normal_part[province_mask].copy()
                df_normal_part.drop(df_normal_part[province_mask].index, inplace=True)
                if area in province_rows['station'].values:
                    df_province = province_rows
                    self.logger.info(f"提取 province/unknown 行: shape={df_province.shape}")
                else:
                    self.logger.warning(f"province 行未通过检查: {area}")
            else:
                self.logger.warning(f"province 行未找到: {area}")

            df_res_list = []
            # 正常数据-MoE v3.1 入口
            if df_normal_part.shape[0] > 0 and self.only_power is False:
                self.logger.info(f"站级推理开始，checkpoint: {self.checkpoints_dir}")
                config_copy = copy.deepcopy(self.config_dict)
                config_copy["data_check"]["future_len"] = self.predict_seq_lens
                result_df_normal_part = multi_station_inference(
                    ds_dataframe=df_normal_part,
                    df_plants_info=None,
                    checkpoints_dir=self.checkpoints_dir,
                    forecasting_type=self.predict_type,
                    config=config_copy,
                )
                df_res_list.append(result_df_normal_part)
                self.logger.info(f"======站级推理完成======")

            # 省级超短期：TabM与MoE都预测，TabM缺失点由MoE补齐。
            if self.predict_type == "ultra_short" and df_province.shape[0] > 0:
                tabm_result = None
                try:
                    self.logger.info("TabM省级推理开始, checkpoint: xxx")
                    tabm_result = ultra_province_predict(
                        ckpt_path = "xxx",
                        data = df.loc[~df["station"].eq("unknown")].copy(),
                        config = "xxx",
                    )
                except Exception as error:
                    self.logger.exception(f"TabM省级推理失败，将使用MoE补充：{error}")

                moe_result = None
                try:
                    self.logger.info(f"MoE省级推理开始, checkpoint: {self.province_checkpoints_dir}")
                    config_province = copy.deepcopy(self.config_dict)
                    config_province["data_check"]["future_len"] = self.predict_seq_lens
                    moe_result = multi_station_inference(
                        ds_dataframe=df_province,
                        df_plants_info=None,
                        checkpoints_dir=self.province_checkpoints_dir,
                        forecasting_type=self.predict_type,
                        config=config_province,
                    )
                except Exception as error:
                    self.logger.exception(f"MoE省级推理失败：{error}")

                result_df_province = self.merge_province_predictions(
                    area,
                    tabm_result,
                    moe_result,
                )
                df_res_list.append(result_df_province)
                self.logger.info(f"======省级推理完成======")

            # 异常数据-tsfm入口
            if df_abnormal_part.shape[0] == 1:  # 只有一行数据会报错，复制一行，但station不能相同
                df_abnormal_part = pd.concat([df_abnormal_part, df_abnormal_part], ignore_index=True)
                df_abnormal_part.loc[1, "station"] = "unknow"
            if df_abnormal_part.shape[0] > 0:
                self.config_dict["backstop_strategy"]["tsfm_only"] = True
                self.config_dict["pre_trained_model"]["future_len"] = self.predict_seq_lens
                self.config_dict["data_check"]["future_len"] = self.predict_seq_lens
                self.logger.info(f"动态修改config：{self.config_dict}")
                result_df_abnormal_part = multi_station_inference(
                    ds_dataframe=df_abnormal_part,
                    df_plants_info=None,
                    checkpoints_dir=self.checkpoints_dir,
                    forecasting_type=self.predict_type,
                    config=self.config_dict,
                )
                df_res_list.append(result_df_abnormal_part)

            # 合并结果（支持多个结果依次 merge）
            if len(df_res_list) > 1:
                result_df = df_res_list[0]
                for _r in df_res_list[1:]:
                    result_df = pd.merge(result_df, _r, on='dtime', how='outer')
            elif len(df_res_list) == 1:
                result_df = df_res_list[0]
            else:
                result_df = pd.DataFrame()
                self.logger.warning("MoE V3.1和tsf均未推理出结果！")

            value_cols = result_df.columns.difference(['dtime'])
            result_df[value_cols] = result_df[value_cols].clip(lower=0)  # 负值置零

            result_df = result_df[
                (result_df['dtime'] >= self.result_s_time) & (result_df['dtime'] <= self.result_e_time)]
            # 夜间结果置零
            result_df = zero_night_hours(result_df)

            if self.predict_type == "short":
                date_str = datetime.datetime.strptime(self.report_time, '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d')
            else:
                date_str = datetime.datetime.strptime(self.report_time, '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M')

            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

            save_full_path = os.path.join(self.save_path, f"hw_nuoya_{date_str}_{self.predict_type}_{area}.parquet")

            result_df.to_parquet(save_full_path)
            self.logger.info(f"结果保存成功：{save_full_path}, shape: {result_df.shape}")

            self.logger.info(f"multi_infer: finish handle... cost time: {time.time() - start_t}")


def zero_night_hours(
        df: pd.DataFrame,
        start_time: str = "20:00",
        end_time: str = "06:00",
        dtime_col: str = "dtime",
) -> pd.DataFrame:
    """
    将 DataFrame 中指定夜间时段的预测值置零。
    时间精度到分钟，支持 15 分钟频率的精确边界控制。

    Args:
        df: 输入 DataFrame，需包含时间列
        start_time: 夜间开始时刻，格式 HH:MM（含），默认 "20:00"
        end_time: 夜间结束时刻，格式 HH:MM（含），默认 "06:00"
        dtime_col: 时间列名，默认 'dtime'

    Returns:
        处理后的 DataFrame（已置零夜间预测值）
    """
    if dtime_col not in df.columns:
        raise ValueError(f"时间列 '{dtime_col}' 不存在，现有列: {list(df.columns)}")

    df = df.copy()
    time_str = df[dtime_col].dt.strftime('%H:%M')
    if start_time <= end_time:
        mask = (time_str >= start_time) & (time_str <= end_time)
    else:
        mask = (time_str >= start_time) | (time_str <= end_time)

    pred_cols = [c for c in df.columns if c != dtime_col]
    zeros = int(mask.sum())
    if zeros > 0:
        df.loc[mask, pred_cols] = 0.0
        multi_logger.info(f"夜间置零: {zeros}/{len(df)} ({zeros / len(df) * 100:.1f}%)")
    else:
        multi_logger.info(f"夜间置零: 0/{len(df)} (0.0%) - 该时间范围内无夜间数据")

    multi_logger.info(f"夜间时段: {start_time} ~ {end_time}")
    multi_logger.info(f"影响列数: {len(pred_cols)} 个预测列")
    return df


def save_ds_data_cache(df, report_time, area, predict_type, base_save_path):
    report_dt = datetime.datetime.strptime(report_time, '%Y-%m-%d %H:%M:%S')
    date_str = report_dt.strftime('%Y-%m-%d')
    time_str = report_dt.strftime('%H:%M')
    dir_path = os.path.join(
        base_save_path, predict_type,
        f"area={area}", f"date={date_str}", f"time={time_str}",
    )
    os.makedirs(dir_path, exist_ok=True)
    date_full_str = datetime.datetime.strptime(report_time, '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M')
    save_path = os.path.join(dir_path, f"hw_nuoya_ds_{date_full_str}_{predict_type}_{area}.parquet")
    df.to_parquet(save_path, index=False)
    multi_logger.info(f"[{report_time}] ✓ DS数据保存成功: {save_path}, shape={df.shape}")


if __name__ == '__main__':
    full_data_path = {}

    report_time_str = '2026-06-30 10:00:00'
    report_time_list = [
        '2026-07-14 11:00:00', '2026-07-14 11:15:00', '2026-07-14 11:30:00', '2026-07-14 11:45:00',
        '2026-07-14 12:00:00', '2026-07-14 12:15:00', '2026-07-14 12:30:00', '2026-07-14 12:45:00',
        '2026-07-14 13:00:00', '2026-07-14 13:15:00', '2026-07-14 13:30:00', '2026-07-14 13:45:00',
        '2026-07-14 14:00:00', '2026-07-14 14:15:00', '2026-07-14 14:30:00', '2026-07-14 14:45:00',
    ]

    forecasting_type = "ultra_short"

    if forecasting_type == "ultra_short":
        for t in report_time_list:
            MultiInfer(
                report_time=t,
                data_path=full_data_path,
                predict_type=forecasting_type
            ).handle()
            # raise ValueError("STOP!")
    elif forecasting_type == "short":
        MultiInfer(
            report_time=report_time_str,
            data_path=full_data_path,
            predict_type=forecasting_type
        ).handle()
