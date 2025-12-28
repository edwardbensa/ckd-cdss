"""Preprocess Hidden CKD data for model training."""

# Imports
import joblib
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from src.config import TABLES_DIR, INTERIM_DATA_DIR, MODELS_DIR

# Load dataset
df = pd.read_csv(TABLES_DIR / "hiddenckd_01.csv")
logger.info(f"Loaded dataset with shape: {df.shape}")


# Define preprocessing
num_features = ["age", "height_cm", "weight_kg", "s_bp", "d_bp", "bmi"]
num_transformer = Pipeline([
    ("power_transform", PowerTransformer(method="yeo-johnson"))
])

nom_features = ["s_eth", "family_kd", "device"]
nom_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

acr_features = ["screening_acr"]
acr_categories = [["<30", "3-30", ">30"]]
acr_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ordinal", OrdinalEncoder(categories=acr_categories))
])

bin_features = ["male", "htn", "dm", "cvd"]
bin_categories = [[False, True] for _ in bin_features]
bin_transformer = Pipeline([
    ("bin_enc", OrdinalEncoder(categories=bin_categories))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_features),
        ("nom", nom_transformer, nom_features),
        ("acr", acr_transformer, acr_features),
        ("bin", bin_transformer, bin_features),
    ]
)


# Split raw data
X = df[num_features + nom_features + acr_features + bin_features]
y = df["ckd_status"]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
logger.info(f"Train shape: {X_train_raw.shape}, Test shape: {X_test_raw.shape}")


# Transform train set
logger.info("Fitting preprocessing on training data...")
X_train = preprocessor.fit_transform(X_train_raw, y_train) # type: ignore
logger.info(f"Class distribution after resampling:\n{y_train.value_counts()}")

# Transform test set
X_test = preprocessor.transform(X_test_raw)
logger.info(f"Final test shape: {X_test.shape}")

# Extract feature names
feature_names = preprocessor.get_feature_names_out().tolist()
logger.info(f"Extracted {len(feature_names)} feature names.")

# Create and save dataframes to CSV
train_df = pd.DataFrame(X_train, columns=feature_names) # type: ignore
train_df["ckd_status"] = y_train.values
train_df["ckd_status"] = train_df["ckd_status"].replace({"ckd": 1, "notckd": 0})
train_df.to_csv(INTERIM_DATA_DIR / 'tabular_data/hiddenckd_train.csv', index=False)

test_df = pd.DataFrame(X_test, columns=feature_names) # type: ignore
test_df["ckd_status"] = y_test.values
test_df["ckd_status"] = test_df["ckd_status"].replace({"ckd": 1, "notckd": 0})
test_df.to_csv(INTERIM_DATA_DIR / 'tabular_data/hiddenckd_test.csv', index=False)

# Save preprocessing pipeline
joblib.dump(preprocessor, MODELS_DIR / "hiddenckd/preprocessor.joblib")
