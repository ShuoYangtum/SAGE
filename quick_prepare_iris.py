from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def main():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["target"] = df["target"].map({idx: name for idx, name in enumerate(iris.target_names)})

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["target"],
    )

    train_path = data_dir / "iris_train.csv"
    test_path = data_dir / "iris_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"[prepare] wrote {train_path} ({len(train_df)} rows)")
    print(f"[prepare] wrote {test_path} ({len(test_df)} rows)")


if __name__ == "__main__":
    main()
