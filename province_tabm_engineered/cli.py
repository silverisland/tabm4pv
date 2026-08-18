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
    test_parser.add_argument("--data")

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
        print(metrics.to_string(index=False))
        print(f"delivery rows returned: {len(predictions):,}", flush=True)
    else:
        result = predict(args.checkpoint, args.data, args.config)
        output_path = Path(args.output).expanduser().resolve()
        result.to_parquet(output_path, index=False)
        print(f"saved {len(result):,} predictions to {output_path}", flush=True)


if __name__ == "__main__":
    main()
