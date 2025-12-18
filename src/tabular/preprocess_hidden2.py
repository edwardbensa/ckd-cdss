"""Preprocess Hidden CKD data for model training with SDV augmentation."""

# Imports
import warnings
import joblib
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata

from src.config import TABLES_DIR, INTERIM_DATA_DIR, MODELS_DIR

INTERIM_TABLES_DIR = INTERIM_DATA_DIR / "tabular_data"
warnings.filterwarnings("ignore", category=FutureWarning)

# Load dataset
df = pd.read_csv(TABLES_DIR / "hiddenckd_01.csv")
logger.info(f"Loaded dataset with shape: {df.shape}")

# Define preprocessing
num_features = ["age", "height_cm", "weight_kg", "systolic", "diastolic"]
num_transformer = Pipeline([
    ("power_transform", PowerTransformer(method="yeo-johnson"))
])

nom_features = ["s_ethnicity", "family_kd", "screening_acr", "device_used"]
nom_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

bin_features = ["male", "has_htn", "has_dm", "has_cvd"]
bin_categories = [[False, True] for _ in bin_features]
bin_transformer = Pipeline([
    ("ord_enc", OrdinalEncoder(categories=bin_categories))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_features),
        ("nom", nom_transformer, nom_features),
        ("bin", bin_transformer, bin_features),
    ]
)

# Split raw data
X = df[num_features + nom_features + bin_features]
y = df["ckd_status"]

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
synthetic_df = synthesizer.sample(num_rows=2000)
logger.info(f"Generated synthetic dataset with shape: {synthetic_df.shape}")

# Concatenate synthetic + real training data
augmented_df = pd.concat([train_df, synthetic_df], ignore_index=True)
X_train_aug = augmented_df[num_features + nom_features + bin_features]
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

# Map y labels
pd.set_option('future.no_silent_downcasting', True)
y_resampled = y_resampled.replace({"CKD": 1, "No CKD": 0})
y_test = y_test.replace({"CKD": 1, "No CKD": 0})
logger.info("Label mapping complete.")

# Save augmented train/test sets
train_df_out = pd.DataFrame(X_resampled, columns=feature_names)
train_df_out["ckd_status"] = y_resampled.values
train_df_out.to_csv(INTERIM_TABLES_DIR / 'hiddenckd_train.csv', index=False)

test_df_out = pd.DataFrame(X_test, columns=feature_names) # type: ignore
test_df_out["ckd_status"] = y_test.values
test_df_out.to_csv(INTERIM_TABLES_DIR / 'hiddenckd_test.csv', index=False)

# Save preprocessing pipeline
joblib.dump(preprocessor, MODELS_DIR / "hiddenckd/preprocessor.joblib")
