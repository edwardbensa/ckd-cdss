"""Preprocess Hidden CKD data for model training with SDV augmentation."""

# Imports
import joblib
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata

from src.config import TABLES_DIR, INTERIM_DATA_DIR, MODELS_DIR

# Load dataset
df = pd.read_csv(TABLES_DIR / "ucickd.csv")
logger.info(f"Loaded dataset with shape: {df.shape}")


# Define preprocessing
num_features = ["age", "d_bp", "sg", "al", "su", "bgr", "bu",
                "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc"]
num_transformer = Pipeline([
    ("power_transform", RobustScaler())
])

bin_features = ["htn", "dm", "cad", "pe", "ane",
                "rbc", "pc", "pcc", "ba", "appet_poor"]
bin_categories = [[False, True] for _ in bin_features]
bin_transformer = Pipeline([
    ("ord_enc", OrdinalEncoder(categories=bin_categories))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_features),
        ("bin", bin_transformer, bin_features),
    ]
)


# Split raw data
X = df[num_features + bin_features]
y = df["class"]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
logger.info(f"Train shape: {X_train_raw.shape}, Test shape: {X_test_raw.shape}")

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
logger.info(f"Train shape: {X_train_raw.shape}, Test shape: {X_test_raw.shape}")

# -------------------------------
# SDV augmentation step
# -------------------------------
logger.info("Training TVAESynthesizer on training data...")
train_df = X_train_raw.copy()
train_df["ckd_status"] = y_train.values

# Fit synthesizer
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=train_df)

synthesizer = TVAESynthesizer(metadata, epochs=300, batch_size=64)
synthesizer.fit(train_df)

# Generate synthetic samples (e.g., 2000 new rows)
synthetic_df = synthesizer.sample(num_rows=1000)
logger.info(f"Generated synthetic dataset with shape: {synthetic_df.shape}")

# Concatenate synthetic + real training data
augmented_df = pd.concat([train_df, synthetic_df], ignore_index=True)
X_train_aug = augmented_df[num_features + bin_features]
y_train_aug = augmented_df["ckd_status"]

# -------------------------------
# Build ImbLearn pipeline
# -------------------------------
resampling_pipeline = ImbPipeline(steps=[
    ("preprocess", preprocessor),
    ("smote_tomek", SMOTEENN(random_state=42))
])

# Transform augmented train set
logger.info("Fitting preprocessing + SMOTEENN on augmented training data...")
X_resampled, y_resampled = resampling_pipeline.fit_resample(X_train_aug, y_train_aug)  # type: ignore
logger.info(f"Class distribution after resampling:\n{y_resampled.value_counts()}")

# Transform test set
preprocessor = resampling_pipeline.named_steps["preprocess"]
X_test = preprocessor.transform(X_test_raw)
logger.info(f"Final test shape: {X_test.shape}")

# Extract feature names
feature_names = resampling_pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
logger.info(f"Extracted {len(feature_names)} feature names.")

# Create and save dataframes to CSV
train_df = pd.DataFrame(X_resampled, columns=feature_names) # type: ignore
train_df["ckd_status"] = y_resampled.values
train_df["ckd_status"] = train_df["ckd_status"].replace({"ckd": 1, "notckd": 0})
train_df.to_csv(INTERIM_DATA_DIR / 'tabular_data/ucickd_train.csv', index=False)

test_df = pd.DataFrame(X_test, columns=feature_names) # type: ignore
test_df["ckd_status"] = y_test.values
test_df["ckd_status"] = test_df["ckd_status"].replace({"ckd": 1, "notckd": 0})
test_df.to_csv(INTERIM_DATA_DIR / 'tabular_data/ucickd_test.csv', index=False)

# Save preprocessing pipeline
joblib.dump(preprocessor, MODELS_DIR / "ucickd/preprocessor.joblib")
