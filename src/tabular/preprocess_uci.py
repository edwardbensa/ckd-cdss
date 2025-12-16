"""Preprocess Hidden CKD data for model training."""

# Imports
import warnings
import joblib
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from src.config import TABLES_DIR, INTERIM_DATA_DIR, MODELS_DIR

INTERIM_TABLES_DIR = INTERIM_DATA_DIR / "tabular_data"
warnings.filterwarnings("ignore", category=FutureWarning)

# Load dataset
df = pd.read_csv(TABLES_DIR / "ucickd.csv")
logger.info(f"Loaded dataset with shape: {df.shape}")


# Define preprocessing
num_features = ["age", "bp", "sg", "al", "su", "bgr", "bu",
                "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc"]
num_transformer = Pipeline([
    ("power_transform", RobustScaler())
])

bin_features = ["htn", "dm", "cad", "pe", "ane",
                "rbc_ab", "pc_ab", "pcc_pr", "ba_pr", "appet_poor"]
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


# Build ImbLearn pipeline
resampling_pipeline = ImbPipeline(steps=[
    ("preprocess", preprocessor),
    ("smote_tomek", SMOTETomek(random_state=42))
])

# Transform train set
logger.info("Fitting preprocessing + SMOTETomek on training data...")
X_resampled, y_resampled = resampling_pipeline.fit_resample(X_train_raw, y_train) # type: ignore
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
y_resampled = y_resampled.replace({"ckd": 1, "notckd": 0})
y_test = y_test.replace({"ckd": 1, "notckd": 0})
logger.info("Label mapping complete.")

# Create and save dataframes to CSV
train_df = pd.DataFrame(X_resampled, columns=feature_names)
train_df["ckd_status"] = y_resampled
train_df.to_csv(INTERIM_TABLES_DIR / 'ucickd_train.csv', index=False)

test_df = pd.DataFrame(X_test, columns=feature_names) # type: ignore
train_df["ckd_status"] = y_test
test_df.to_csv(INTERIM_TABLES_DIR / 'ucickd_test.csv', index=False)

# Save preprocessing pipeline
joblib.dump(preprocessor, MODELS_DIR / "ucimlrepo_ckd/preprocessor.joblib")
