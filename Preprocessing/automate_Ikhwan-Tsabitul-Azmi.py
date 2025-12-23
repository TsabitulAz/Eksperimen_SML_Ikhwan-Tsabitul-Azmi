import pandas as pd
import argparse
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from joblib import dump


def preprocess(
    input_path: str,
    output_dir: str,
    target_column: str,
    test_size: float = 0.3,
    random_state: int = 42,
):
    # --- Validation ---
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    df = pd.read_csv(input_path)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # --- Split features & target ---
    X = df.drop(columns=[target_column])
    y = df[target_column]

    categorical_features = X.columns.tolist()

    # --- Preprocessing pipeline ---
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OrdinalEncoder(), categorical_features)
        ]
    )

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # --- Train / Test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )

    # --- Transform ---
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    os.makedirs(output_dir, exist_ok=True)

    # --- Save processed datasets ---
    X_train_df = pd.DataFrame(
        X_train_processed,
        columns=categorical_features
    )
    X_test_df = pd.DataFrame(
        X_test_processed,
        columns=categorical_features
    )

    y_train_df = pd.DataFrame(
        {target_column: y_train}
    )
    y_test_df = pd.DataFrame(
        {target_column: y_test}
    )

    # Combined datasets (CI-friendly)
    train_final = pd.concat([X_train_df, y_train_df], axis=1)
    test_final = pd.concat([X_test_df, y_test_df], axis=1)

    train_final.to_csv(
        os.path.join(output_dir, "train_processed.csv"),
        index=False
    )
    test_final.to_csv(
        os.path.join(output_dir, "test_processed.csv"),
        index=False
    )

    # --- Save artifacts ---
    dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))
    dump(label_encoder, os.path.join(output_dir, "label_encoder.joblib"))

    # --- CI log ---
    print("Preprocessing finished successfully")
    print(f"Input        : {input_path}")
    print(f"Output dir   : {output_dir}")
    print(f"Train shape  : {train_final.shape}")
    print(f"Test shape   : {test_final.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CI-friendly preprocessing entrypoint"
    )

    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--target_column", type=str, default="class")
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()

    preprocess(
        input_path=args.input_path,
        output_dir=args.output_dir,
        target_column=args.target_column,
        test_size=args.test_size,
        random_state=args.random_state,
    )
