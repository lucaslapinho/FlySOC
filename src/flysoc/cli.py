"""Small CLI for local, file-based research workflows."""

import argparse
import json
import logging
from pathlib import Path

from .artifacts import json_value, run_id, sha256, write_json
from .config import ROOT, load_config
from .synthetic import generate_alerts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="FlySOC local research lab")
    commands = parser.add_subparsers(dest="command", required=True)
    generator = commands.add_parser("generate", help="Generate inert synthetic alert records")
    generator.add_argument("--config", type=Path)
    generator.add_argument("--size", type=int)
    train = commands.add_parser("train", help="Train on the oldest chronological partition")
    train.add_argument("--config", type=Path)
    train.add_argument("--data", type=Path)
    evaluate = commands.add_parser("evaluate", help="Evaluate saved chronological splits")
    evaluate.add_argument("--run", help="Saved run identifier; default latest run")
    inspect = commands.add_parser("inspect", help="Inspect a saved alert against training history")
    inspect.add_argument("--run")
    inspect.add_argument("--alert-id", required=True)
    inspect.add_argument("--k", type=int, default=5)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.command == "generate":
            config = load_config(args.config)
            frame = generate_alerts(config, args.size)
            directory = ROOT / "data/synthetic"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{run_id('synthetic')}.csv"
            frame.to_csv(path, index=False)
            distribution = {"path": str(path), "sha256": sha256(path), "size": len(frame),
                            "seed": config["seed"], "configuration": config,
                            "family_counts": frame.family.value_counts().to_dict(),
                            "verdict_counts": frame.verdict.value_counts().to_dict(),
                            "unseen_count": int(frame.is_unseen.sum())}
            write_json(path.with_suffix(".json"), distribution)
            write_json(directory / "latest.json", {"filename": path.name})
            print(json.dumps(distribution, indent=2))
        elif args.command == "train":
            from .experiment import train_experiment
            dataset = args.data
            if dataset is None:
                latest = json.loads((ROOT / "data/synthetic/latest.json").read_text(encoding="utf-8"))["filename"]
                if Path(latest).name != latest:
                    raise ValueError("Unsafe dataset pointer")
                dataset = ROOT / "data/synthetic" / latest
            print(train_experiment(dataset, args.config))
        elif args.command == "evaluate":
            from .evaluation import evaluate_experiment
            print(evaluate_experiment(args.run))
        elif args.command == "inspect":
            from .experiment import load_experiment
            _, payload, splits = load_experiment(args.run)
            import pandas as pd
            alerts = pd.concat(splits.values(), ignore_index=True)
            selected = alerts.loc[alerts.alert_id == args.alert_id]
            if len(selected) != 1:
                raise ValueError("Alert ID not found uniquely in the saved splits")
            result = payload["pipeline"].inspect(selected, args.k)[0]
            result["matches_include_self"] = args.alert_id in set(splits["train"].alert_id)
            result["historical_context_note"] = "Historical labels are context, not a confirmed verdict for the query. Training alerts may match themselves."
            model = payload["classifiers"].get("fly_logistic_regression")
            if model is not None:
                probabilities = model.predict_proba(payload["pipeline"].encode(selected))[0]
                result["downstream_classifier"] = {"name": "fly_logistic_regression", "prediction": model.classes_[probabilities.argmax()],
                                                     "probabilities": dict(zip(model.classes_, probabilities))}
            print(json.dumps(json_value(result), indent=2, allow_nan=False))
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.exit(2, f"FlySOC error: {error}\n")
