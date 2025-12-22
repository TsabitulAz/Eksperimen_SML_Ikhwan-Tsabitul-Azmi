import pandas as pd
import argparse
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from joblib import dump


def preprocess_car_evaluation(
    input_path: str,
    output_dir: str,
    target_column: str = "class"
):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset not found at {input_path}")

    df = pd.read_csv(input_path)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    categorical_features = X.columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(), categorical_features)
        ]
    )

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.3,
        random_state=42,
        stratify=y_encoded
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame(
        X_train_processed,
        columns=categorical_features
    ).to_csv(f"{output_dir}/X_train_processed.csv", index=False)

    pd.DataFrame(
        X_test_processed,
        columns=categorical_features
    ).to_csv(f"{output_dir}/X_test_processed.csv", index=False)

    pd.Series(y_train, name="target").to_csv(
        f"{output_dir}/y_train.csv", index=False
    )

    pd.Series(y_test, name="target").to_csv(
        f"{output_dir}/y_test.csv", index=False
    )

    dump(preprocessor, f"{output_dir}/preprocessor_pipeline.joblib")
    dump(label_encoder, f"{output_dir}/label_encoder.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CI-friendly preprocessing pipeline")

    parser.add_argument(
        "--input_path",
        type=str,
        default="dataset_raw/car_evaluation.csv"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Preprocessing/dataset_preprocessing"
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="class"
    )

    args = parser.parse_args()

    preprocess_car_evaluation(
        input_path=args.input_path,
        output_dir=args.output_dir,
        target_column=args.target_column
    )
