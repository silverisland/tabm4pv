from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill

from .data import Dataset
from .metrics import Evaluation
from .models import ModelResult
from .split import SplitPlan


def _style(worksheet) -> None:
    fill = PatternFill("solid", fgColor="1F4E78")
    for cell in worksheet[1]:
        cell.fill = fill
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center")
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
    for cells in worksheet.columns:
        width = min(42, max(10, max(len(str(cell.value or "")) for cell in cells) + 2))
        worksheet.column_dimensions[cells[0].column_letter].width = width


def write_report(
    path: Path,
    dataset: Dataset,
    split: SplitPlan,
    result: ModelResult,
    evaluation: Evaluation,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([
        {"项目": "实验名称", "值": config["experiment"]["name"]},
        {"项目": "站点", "值": dataset.station},
        {"项目": "起始时间", "值": dataset.origins.min()},
        {"项目": "结束时间", "值": dataset.origins.max()},
        {"项目": "样本数", "值": len(dataset.frame)},
        {"项目": "装机容量最小值", "值": float(dataset.capacity.min())},
        {"项目": "装机容量最大值", "值": float(dataset.capacity.max())},
        {"项目": "气象字段", "值": ", ".join(dataset.weather_columns)},
        {"项目": "组件模型", "值": ", ".join(result.predictions)},
        {"项目": "真值策略", "值": "同一目标时刻统一使用Index0真值"},
    ])
    sheets = {
        "实验摘要": summary,
        "数据质量": dataset.quality,
        "真值冲突": dataset.target_conflicts,
        "数据切分": split.summary,
        "逐点指标": evaluation.horizon_metrics,
        "分段指标": evaluation.segment_metrics,
        "场景指标": evaluation.regime_metrics,
        "曲线指标": evaluation.curve_metrics,
        "每日指标": evaluation.daily_metrics,
        "月度指标": evaluation.monthly_metrics,
        "融合对比": evaluation.fusion_comparison,
        "融合权重": result.fusion_weights,
    }
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            (frame if not frame.empty else pd.DataFrame({"说明": ["无可用结果"]})).to_excel(
                writer, sheet_name=name, index=False
            )
        for worksheet in writer.book.worksheets:
            _style(worksheet)
