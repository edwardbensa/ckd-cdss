"""Script to load and preprocess CKD tabular datasets."""

# Imports
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from src.config import RAW_TABLES_DIR, TABLES_DIR
from src.tabular.utils.dataset_utils import (
    compute_bmi, simplify_ethnicity, black_ancestry,
    impute_acr, gfr_staging, acr_staging, ckd_status)

df1 = pd.read_csv(RAW_TABLES_DIR / 'hiddenckd.csv')

# Select relevant columns
df1 = df1[[
    "PID", "Gender", "D.O.B.", "Date of Event",
    "Ethnicity", "Height (cm)", "Weight(kg)", "Mean Systolic BP", "Mean Diastolic BP",
    "Medical Conditions (Summary)", "Pre-existing conditions ", "Family history of CKD?",
    "Current Medications?", "Which device was used?",
    "uACR category", "Lab uACR", "eGFR", "True positive or negative"
]]

# Rename columns
df1.columns = [col.lower().strip() for col in df1.columns]
df1 = df1.rename(columns={
    "d.o.b.": "dob",
    "date of event": "event_date",
    "height (cm)": "height_cm",
    "weight(kg)": "weight_kg",
    "ethnicity": "eth", 
    "mean systolic bp": "s_bp",
    "mean diastolic bp": "d_bp",
    "medical conditions (summary)": "medical_conditions",
    "pre-existing conditions": "disclosed_conditions",
    "family history of ckd?": "family_kd",
    "current medications?": "current_medications",
    "which device was used?": "device",
    "true positive or negative": "confirmed_albuminuria",
    "uacr category": "screening_acr",
    "lab uacr": "acr",
    "egfr": "egfr"
})

# Replace values
df1["egfr"] = df1["egfr"].replace({">": "", "<": ""}, regex=True).astype(float)
df1["screening_acr"] = df1["screening_acr"].replace({" ": ""}, regex=True)
df1["screening_acr"] = df1["screening_acr"].apply(lambda x: "3-30" if len(x)>3 else x)
df1["confirmed_albuminuria"] = df1["confirmed_albuminuria"].map({"True positive": True,
                                                                 "False positive": False})
df1["family_kd"] = df1["family_kd"].replace({"Definitely yes": "yes",
                                             "Definitely not": "no",
                                             "Not sure": "unsure"})
df1['eth'] = df1['eth'].str.strip().replace(
    {'Black African' : 'Black African (unspecified)'})
df1["device"] = df1["device"].str.strip().replace({
    "Siemens device used": "Siemens Clinitek",
})

# Combine 'medical_conditions' and 'disclosed_conditions' into 'comorbidities'
df1["comorbidities"] = df1.apply(
    lambda row: f"{row['medical_conditions']}; {row['disclosed_conditions']}",
    axis=1
)

# Calculated columns
df1["dob"] = pd.to_datetime(df1["dob"], dayfirst = True)
df1["event_date"] = pd.to_datetime(df1["event_date"], dayfirst = True)
df1["age"] = ((df1["event_date"] - df1["dob"]).dt.days / 365).round(1)  # type: ignore

df1["htn"] = df1["comorbidities"].str.contains("HTN", na=False)
df1["dm"] = df1["comorbidities"].str.contains("DM", na=False)
df1["cvd"] = df1["comorbidities"].str.contains("Heart", na=False)

df1["bp_meds"] = df1["current_medications"].str.contains("Blood pressure", na=False)
df1["dm_meds"] = df1["current_medications"].str.contains("Diabetes", na=False)
df1["chol_meds"] = df1["current_medications"].str.contains("Cholesterol", na=False)

df1["male"] = df1["gender"].map({"Male": True, "Female": False})

df1["bmi"] = compute_bmi(df1["weight_kg"], df1["height_cm"])

df1["s_eth"] = simplify_ethnicity(df1["eth"])
df1["eth_black"] = black_ancestry(df1["eth"])

# Copy dataframe and filter undiagnosed
df2 = df1.copy()
df2 = df2[df2["egfr"].isna()]
df2 = df2.reset_index(drop=True)

# Fiilter out rows
df1 = df1[df1["egfr"].notna()]
df1 = df1.reset_index(drop=True)

# ACR imputation
# Impute missing ACR values if confirmed_albuminuria is False
df1["acr"] = df1.apply(impute_acr, axis=1)

# Impute remaining missing values using KNN imputer
df1_imputed = df1.copy()

# Encode categorical variables
categorical_cols = ['s_eth', 'family_kd', 'screening_acr']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df1_imputed[col] = le.fit_transform(df1_imputed[col].astype(str))
    label_encoders[col] = le

# Convert booleans to int
bool_cols = ['male', 'htn', 'dm', 'cvd']
df1_imputed[bool_cols] = df1_imputed[bool_cols].astype(int)

# Select predictors for imputation
impute_cols = [
    'acr', 'egfr', 'age', 'height_cm', 'weight_kg',
    's_bp', 'd_bp', 'male', 'htn', 'dm', 'cvd',
    's_eth', 'family_kd'
]

data_for_impute = df1_imputed[impute_cols]

# Scale data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_for_impute)

# Apply KNN Imputer and reverse scaling
imputer = KNNImputer(n_neighbors=5, weights="uniform")
imputed_array = imputer.fit_transform(data_scaled)
imputed_array = scaler.inverse_transform(imputed_array)

# Replace imputed values back into df and round acr
df1_imputed[impute_cols] = imputed_array
df1_imputed['acr'] = df1_imputed['acr'].round(1)

# Update original df with imputed ACR values
df1["acr"] = df1_imputed["acr"]


# Use egfr and acr to determine ckd status and stage
df1["gfr_stage"] = df1.apply(gfr_staging, axis=1)
df1["acr_stage"] = df1.apply(acr_staging, axis=1)
df1["ckd_status"] = df1.apply(ckd_status, axis=1)

# Final column selection
df1 = df1[[
    "male", "eth", "s_eth", "eth_black", "dob", "age", "height_cm", "weight_kg", "bmi",
    "s_bp", "d_bp", "htn", "dm", "cvd", "bp_meds", "dm_meds", "chol_meds", "family_kd", "device",
    "screening_acr", "confirmed_albuminuria", "acr", "egfr",
    "gfr_stage", "acr_stage", "ckd_status"
]]

df2 = df2[[
    "male", "eth", "s_eth", "eth_black", "dob", "age", "height_cm", "weight_kg", "bmi",
    "s_bp", "d_bp", "htn", "dm", "cvd", "bp_meds", "dm_meds", "chol_meds",
    "family_kd", "device", "screening_acr"
]]

# Save processed dataset
df1.to_csv(TABLES_DIR / 'hiddenckd_01.csv', index=False)
df2.to_csv(TABLES_DIR / 'hiddenckd_02.csv', index=False)
