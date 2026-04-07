import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import torch

from sage import SAGE


def prepare_iris(data_dir: Path, seed: int):
    data_dir.mkdir(exist_ok=True, parents=True)
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["target"] = df["target"].map({idx: name for idx, name in enumerate(iris.target_names)})
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        stratify=df["target"],
    )
    train_path = data_dir / "iris_train.csv"
    test_path = data_dir / "iris_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_df, test_df, train_path, test_path


def evaluate_downstream(syn_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str):
    x_train = syn_df.drop(columns=[target_col])
    y_train = syn_df[target_col].astype(str)
    x_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].astype(str)

    models = {
        "DT": DecisionTreeClassifier(random_state=42),
        "RF": RandomForestClassifier(n_estimators=300, random_state=42),
    }
    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        results[name] = {
            "acc": float(accuracy_score(y_test, preds)),
            "f1_macro": float(f1_score(y_test, preds, average="macro")),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Formal Iris evaluation for SAGE FS/LC.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mi-threshold", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    data_dir = Path("data")
    target_col = "target"

    train_df, test_df, train_path, _ = prepare_iris(data_dir, seed=args.seed)

    generator = SAGE(
        model_name=args.model_name,
        device=torch.device(args.device),
        use_lora=False,
    )
    generator.fit(
        data=str(train_path),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=128,
        val_ratio=0.1,
        early_stopping_rounds=5,
        mi_n_bins=10,
        mi_strategy="fd",
        mi_threshold=args.mi_threshold,
        constrain_string_values=True,
        num_workers=0,
        gradient_accumulation_steps=1,
        shuffle=True,
        seed=args.seed,
        deterministic=False,
        checkpoint_path=str(output_dir / "iris_best_model.pt"),
    )

    sample_num = len(train_df)
    fs_syn = generator.sample(
        sample_num=sample_num,
        p=0.7,
        temperature=1.0,
        mi_threshold=args.mi_threshold,
        copy_factor=1,
        apply_final_constraints=True,
    )
    lc_syn = generator.sample_logits(
        sample_num=sample_num,
        p=0.7,
        temperature=1.0,
        mi_lambda=1.0,
        scale_clip_min=0.5,
        scale_clip_max=1.5,
        apply_final_constraints=True,
    )

    fs_metrics = evaluate_downstream(fs_syn, test_df, target_col=target_col)
    lc_metrics = evaluate_downstream(lc_syn, test_df, target_col=target_col)

    report = {
        "dataset": "iris",
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "config": {
            "model_name": args.model_name,
            "device": args.device,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "mi_threshold": args.mi_threshold,
        },
        "FS": fs_metrics,
        "LC": lc_metrics,
    }

    report_path = output_dir / "iris_formal_metrics.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=== Iris Formal Evaluation Results ===")
    print(json.dumps(report, indent=2))
    print(f"Saved metrics to: {report_path}")


if __name__ == "__main__":
    main()
