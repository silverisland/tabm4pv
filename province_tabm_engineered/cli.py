from __future__ import annotations

import argparse
from pathlib import Path

from .api import predict, test, train


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="省级超短期 TabM 工程化接口")
    commands = result.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="训练并保存 checkpoint")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--data")

    test_parser = commands.add_parser("test", help="在有标签数据上评估")
    test_parser.add_argument("--config", required=True)
    test_parser.add_argument("--checkpoint", required=True)
    test_parser.add_argument("--data", required=True)
    test_parser.add_argument("--output", default="evaluation_predictions.parquet")

    predict_parser = commands.add_parser("predict", help="生成预测 DataFrame 并保存")
    predict_parser.add_argument("--config", required=True)
    predict_parser.add_argument("--checkpoint", required=True)
    predict_parser.add_argument("--data", required=True)
    predict_parser.add_argument("--output", required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "train":
        result = train(args.config, args.data)
        print(result["metrics"].to_string(index=False))
        print(f"checkpoint: {result['checkpoint_dir']}")
    elif args.command == "test":
        metrics, predictions = test(args.checkpoint, args.data, args.config)
        predictions.to_parquet(args.output, index=False)
        metrics.to_csv(Path(args.output).with_suffix(".metrics.csv"), index=False)
        print(metrics.to_string(index=False))
    else:
        result = predict(args.checkpoint, args.data, args.config)
        result.to_parquet(args.output, index=False)
        print(f"saved {len(result):,} predictions to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
