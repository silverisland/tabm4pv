from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__:
    from .api import predict, test, train
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from province_tabm_engineered.api import predict, test, train


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="省级超短期 TabM")
    commands = result.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="训练模型")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--data")

    test_parser = commands.add_parser("test", help="测试模型")
    test_parser.add_argument("--config", required=True)
    test_parser.add_argument("--checkpoint", required=True)
    test_parser.add_argument("--data")

    predict_parser = commands.add_parser("predict", help="执行单起报时刻推理")
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
        print(f"delivery rows returned: {len(predictions):,}")
    else:
        predictions = predict(args.checkpoint, args.data, args.config)
        output = Path(args.output).expanduser().resolve()
        predictions.to_parquet(output, index=False)
        print(f"saved {len(predictions):,} predictions to {output}")


if __name__ == "__main__":
    main()
