from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from joblib import dump

def preprocess_car_evaluation(data, target_column="class", save_path="preprocessor_pipeline.joblib"):
    # Semua fitur adalah kategorikal (ordinal)
    categorical_features = data.drop(columns=[target_column]).columns.tolist()

    # Ordinal Encoder
    categorical_transformer = Pipeline(steps=[
        ("encoder", OrdinalEncoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    X = data.drop(columns=[target_column])
    y = data[target_column]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.3,
        random_state=42,
        stratify=y_encoded
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    dump(preprocessor, save_path)
    dump(label_encoder, "label_encoder.joblib")

    return X_train_processed, X_test_processed, y_train, y_test