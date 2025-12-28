"""Registry and utility functions for data entry."""

# Imports
from datetime import date
from collections import defaultdict
import streamlit as st
from src.tabular.utils.dataset_utils import compute_bmi, gfr_staging, acr_staging, ckd_status

ethnicities = [
    "White: English, Welsh, Scottish, Northern Irish or British",
    "White: Irish",
    "White: Gypsy or Irish Traveller",
    "White: Roma",
    "White: Any other White background",

    "Mixed or Multiple ethnic groups: White and Black Caribbean",
    "Mixed or Multiple ethnic groups: White and Black African",
    "Mixed or Multiple ethnic groups: White and Asian",
    "Mixed or Multiple ethnic groups: Any other Mixed or Multiple background",

    "Asian or Asian British: Indian",
    "Asian or Asian British: Pakistani",
    "Asian or Asian British: Bangladeshi",
    "Asian or Asian British: Chinese",
    "Asian or Asian British: Any other Asian background",

    "Black, Black British, Caribbean or African: Caribbean",
    "Black, Black British, Caribbean or African: Central African",
    "Black, Black British, Caribbean or African: Eastern African",
    "Black, Black British, Caribbean or African: Northern African",
    "Black, Black British, Caribbean or African: Southern African",
    "Black, Black British, Caribbean or African: West African",
    "Black, Black British, Caribbean or African: Any other Black background",

    "Other ethnic group: Arab",
    "Other ethnic group: Any other ethnic group"
]


FIELD_REGISTRY = {
    # Identifiers and Demographics
    "email": {
        "label": "Email Address",
        "widget": "text",
        "placeholder": "e.g., user@example.com",
        "section": "Identifiers and Demographics"
    },
    "postcode": {
        "label": "Postcode",
        "widget": "text",
        "placeholder": "e.g., BR6 7AA",
        "section": "Identifiers and Demographics"
    },
    "male": {
        "label": "Sex",
        "widget": "select",
        "options": [True, False],
        "format_func": lambda x: "Male" if x else "Female",
        "section": "Identifiers and Demographics"
    },
    "age": {
        "label": "Age",
        "widget": "number",
        "min": 0.0,
        "max": 120.0,
        "default": 50.0,
        "section": "Identifiers and Demographics"
    },
    "dob": {
        "label": "Date of Birth",
        "widget": "date",
        "section": "Identifiers and Demographics"
    },
    "eth": {
        "label": "Ethnicity",
        "widget": "select",
        "options": ethnicities,
        "placeholder": "e.g., White British",
        "section": "Identifiers and Demographics"
    },

    # Anthropometrics
    "height_cm": {
        "label": "Height (cm)",
        "widget": "number",
        "min": 50.0,
        "max": 250.0,
        "default": 170.0,
        "section": "Anthropometrics"
    },
    "weight_kg": {
        "label": "Weight (kg)",
        "widget": "number",
        "min": 10.0,
        "max": 300.0,
        "default": 70.0,
        "section": "Anthropometrics"
    },

    # Blood Pressure
    "s_bp": {
        "label": "Systolic BP (mmHg)",
        "widget": "number",
        "min": 50.0,
        "max": 250.0,
        "default": 120.0,
        "section": "Blood Pressure"
    },
    "d_bp": {
        "label": "Diastolic BP (mmHg)",
        "widget": "number",
        "min": 30.0,
        "max": 150.0,
        "default": 80.0,
        "section": "Blood Pressure"
    },

    # Family History
    "family_htn": {
        "label": "Family History of Hypertension",
        "widget": "select",
        "options": ["yes", "no", "unsure"],
        "section": "Family History"
    },
    "family_dm": {
        "label": "Family History of Diabetes",
        "widget": "select",
        "options": ["yes", "no", "unsure"],
        "section": "Family History"
    },
    "family_kd": {
        "label": "Family History of Kidney Disease",
        "widget": "select",
        "options": ["yes", "no", "unsure"],
        "section": "Family History"
    },

    # Comorbidities
    "htn": {
        "label": "Hypertension",
        "widget": "select",
        "options": [False, True],
        "default": False,
        "format_func": lambda x: "Yes" if x else "No",
        "section": "Comorbidities"
    },
    "dm": {
        "label": "Diabetes Mellitus",
        "widget": "select",
        "options": [False, True],
        "default": False,
        "format_func": lambda x: "Yes" if x else "No",
        "section": "Comorbidities"
    },
    "kd": {
        "label": "Kidney Disease",
        "widget": "select",
        "options": [False, True],
        "default": False,
        "format_func": lambda x: "Yes" if x else "No",
        "section": "Comorbidities"
    },
    "cvd": {
        "label": "Cardiovascular Disease",
        "widget": "select",
        "options": [False, True],
        "default": False,
        "format_func": lambda x: "Yes" if x else "No",
        "section": "Comorbidities"
    },
    "cad": {
        "label": "Coronary Artery Disease",
        "widget": "select",
        "options": [False, True],
        "default": False,
        "format_func": lambda x: "Yes" if x else "No",
        "section": "Comorbidities"
        },

    # Urine Microscopy and Dipstick Analysis
    "rbc": {
        "label": "Red Blood Cells (Abnormal)",
        "widget": "select",
        "options": [False, True],
        "default": False,
        "format_func": lambda x: "Abnormal" if x else "Normal",
        "section": "Urine Microscopy and Dipstick Analysis"
    },
    "pc": {
        "label": "Pus Cells (Abnormal)",
        "widget": "select",
        "options": [False, True],
        "default": False,
        "format_func": lambda x: "Abnormal" if x else "Normal",
        "section": "Urine Microscopy and Dipstick Analysis"
    },
    "pcc": {
        "label": "Pus Cell Clumps",
        "widget": "select",
        "options": [False, True],
        "default": False,
        "format_func": lambda x: "Present" if x else "Not Present",
        "section": "Urine Microscopy and Dipstick Analysis"
    },
    "ba": {
        "label": "Bacteria",
        "widget": "select",
        "options": [False, True],
        "default": False,
        "format_func": lambda x: "Present" if x else "Not Present",
        "section": "Urine Microscopy and Dipstick Analysis"
    },
    "su": {
        "label": "Sugar",
        "widget": "number",
        "min": 0.0,
        "max": 10.0,
        "default": 0.0,
        "section": "Urine Microscopy and Dipstick Analysis"
    },
    "al": {
        "label": "Urine Albumin",
        "widget": "number",
        "min": 0.0,
        "max": 5.0,
        "default": 0.0,
        "section": "Urine Microscopy and Dipstick Analysis"
    },
    "sg": {
        "label": "Specific Gravity",
        "widget": "number",
        "min": 1.0,
        "max": 1.1,
        "default": 1.0,
        "format": "%.3f",
        "section": "Urine Microscopy and Dipstick Analysis"
    },

    # Blood Chemistry and Haematology
    "bgr": {
        "label": "Random Blood Glucose",
        "widget": "number",
        "min": 0.0,
        "max": 40.0,
        "default": 5.0,
        "section": "Blood Chemistry and Haematology"
    },
    "bu": {
        "label": "Blood Urea",
        "widget": "number",
        "min": 0.0,
        "max": 300.0,
        "default": 15.0,
        "section": "Blood Chemistry and Haematology"
    },
    "sc": {
        "label": "Serum Creatinine",
        "widget": "number",
        "min": 0.0,
        "max": 20.0,
        "default": 1.0,
        "section": "Blood Chemistry and Haematology"
    },
    "sod": {
        "label": "Sodium",
        "widget": "number",
        "min": 100.0,
        "max": 200.0,
        "default": 140.0,
        "section": "Blood Chemistry and Haematology"
    },
    "pot": {
        "label": "Potassium",
        "widget": "number",
        "min": 1.0,
        "max": 10.0,
        "default": 4.0,
        "section": "Blood Chemistry and Haematology"
    },
    "hemo": {
        "label": "Hemoglobin",
        "widget": "number",
        "min": 0.0,
        "max": 25.0,
        "default": 12.0,
        "section": "Blood Chemistry and Haematology"
    },
    "pcv": {
        "label": "Packed Cell Volume (%)",
        "widget": "number",
        "min": 0,
        "max": 100,
        "default": 42,
        "section": "Blood Chemistry and Haematology"
    },
    "wbcc": {
        "label": "White Blood Cell Count",
        "widget": "number",
        "min": 0.0,
        "max": 50000.0,
        "default": 7500,
        "section": "Blood Chemistry and Haematology"
    },
    "rbcc": {
        "label": "Red Blood Cell Count",
        "widget": "number",
        "min": 0.0,
        "max": 10.0,
        "default": 4.5,
        "section": "Blood Chemistry and Haematology"
    },

    # Symptoms
    "appet_poor": {
        "label": "Poor Appetite",
        "widget": "select",
        "options": [False, True],
        "default": False,
        "format_func": lambda x: "Poor" if x else "Good",
        "section": "Symptoms"
    },
    "pe": {
        "label": "Pedal Edema",
        "widget": "select",
        "options": [False, True],
        "default": False,
        "format_func": lambda x: "Yes" if x else "No",
        "section": "Symptoms"
    },
    "ane": {
        "label": "Anemia",
        "widget": "select",
        "options": [False, True],
        "default": False,
        "format_func": lambda x: "Yes" if x else "No",
        "section": "Symptoms"
    },

    # ACR/EGFR Diagnostics
    "screening_acr": {
        "label": "Screening ACR Category",
        "widget": "select",
        "options": ["<30", "3-30", ">30"],
        "section": "ACR/EGFR Diagnostics"
    },
    "device": {
        "label": "Device Used",
        "widget": "select",
        "options": ["Healthy.io", "Siemens Clinitek"],
        "section": "ACR/EGFR Diagnostics"
    },
    "acr": {
        "label": "Albumin-Creatinine Ratio (mg/g)",
        "widget": "number",
        "min": 0.0,
        "max": 300.0,
        "default": 0.0,
        "section": "ACR/EGFR Diagnostics"
    },
    "egfr": {
        "label": "eGFR (mL/min/1.73m²)",
        "widget": "number",
        "min": 0.0,
        "max": 200.0,
        "default": 90.0,
        "section": "ACR/EGFR Diagnostics"
    },

    # Computed Fields
    "bmi": {
        "label": "Body Mass Index",
        "widget": "number",
        "computed": True
    },
    "acr_stage": {
        "label": "ACR Stage",
        "widget": "text",
        "computed": True
    },
    "gfr_stage": {
        "label": "eGFR Stage",
        "widget": "text",
        "computed": True
    },
    "ckd_status": {
        "label": "Confirmed CKD Status",
        "widget": "text",
        "computed": True
    }
}

DIPSTICK_TO_FIELDS = {
    "glucose": "su",
    "SG": "sg",
    "leukocytes": "pc",
    "nitrite": "ba",
    "blood": "rbc",
}

def dipstick_results_to_fields(results: dict):
    """Converts dipstick results into fields for field registry."""
    mapped = {}

    for pad_name, info in results.items():
        field = DIPSTICK_TO_FIELDS.get(pad_name)
        if not field:
            continue

        raw_value = info.get("value", "")

        # Extract numeric part if needed
        if "(" in raw_value:
            numeric = raw_value.split("(")[-1].replace(")", "")
        else:
            numeric = raw_value

        # Convert to float where possible
        try:
            numeric = float(numeric)
        except ValueError:
            numeric = raw_value

        mapped[field] = numeric

    return mapped

def convert_dipstick_readings(results: dict):
    """Converts dipstick results into fields for field registry."""
    mapped = {}

    glucose_map = {
        "neg": 0,
        "100(5.5)": 1,
        "250(14)": 2,
        "500(28)": 3,
        "1000(55)": 4
    }

    def convert_sg(val):
        try:
            return float(val)
        except ValueError:
            return None

    def convert_su(val):
        return glucose_map.get(val, None)

    def convert_bool_from_neg(val):
        if val is None:
            return None
        return False if val.lower() == "neg" else True

    def convert_ba(val):
        if val is None:
            return None
        return False if val.lower() in ["neg", "trace"] else True


    for pad_name, info in results.items():
        field = DIPSTICK_TO_FIELDS.get(pad_name)
        if not field:
            continue

        raw_value = info.get("value", "")

        conversion_funcs = {
        "sg": convert_sg(raw_value),
        "su": convert_su(raw_value),
        "rbc": convert_bool_from_neg(raw_value),
        "pc": convert_bool_from_neg(raw_value),
        "ba": convert_ba(raw_value)
        }

        mapped[field] = conversion_funcs[field]

    return mapped


def group_fields_by_section(registry):
    """Group fields by section"""
    sections = defaultdict(list)
    for key, cfg in registry.items():
        section = cfg.get("section", "Identifiers and Demographics")
        sections[section].append((key, cfg))
    return sections


def compute_derived_fields(data):
    """Compute BMI, ACR stage, GFR stage and CKD Status."""
    # BMI
    if data.get("height_cm") and data.get("weight_kg"):
        data["bmi"] = compute_bmi(data["weight_kg"], data["height_cm"])
    else:
        data["bmi"] = None

    # ACR stage
    if data.get("acr") is not None:
        data["acr_stage"] = acr_staging(data)
    else:
        data["acr_stage"] = None

    # eGFR stage
    if data.get("egfr") is not None:
        data["gfr_stage"] = gfr_staging(data)
    else:
        data["gfr_stage"] = None

    # CKD status
    if data.get("gfr_stage") is not None and data.get("acr_stage") is not None:
        data["ckd_status"] = ckd_status(data)
    else:
        data["ckd_status"] = None

    return data


def validate_dates(values):
    """Ensure date is not in the future."""
    errors = []

    # Validate DOB
    dob = values.get("dob")
    if dob and dob > date.today():
        errors.append("Date of birth cannot be in the future.")

    return errors


def numeric_bounds(config):
    """Return (min, max, default, type) for numeric widgets."""
    vals = (config.get("min"), config.get("max"))

    n_type = float if any(isinstance(v, float) for v in vals if v is not None) else int

    def cast(v):
        return None if v is None else n_type(v)

    return cast(vals[0]), cast(vals[1]), n_type



def render_field(key, config, value=None):
    """Render a Streamlit widget based on the registry definition."""
    label = config["label"] + ("*" if config.get("required") else "")

    if config["widget"] == "text":
        return st.text_input(label, value=value, placeholder=config.get("placeholder"))

    min_val, max_val, n_type = numeric_bounds(config)

    if config["widget"] == "number":
        return st.number_input(
            label,
            min_value=min_val,
            max_value=max_val,
            value=n_type(value) if value is not None else None,
            format=config.get("format", None), placeholder="Enter Value",
        )

    if config["widget"] == "select":
        fmt = config.get("format_func")

        if fmt:
            return st.selectbox(
                label,
                config["options"],
                index=config["options"].index(value) if value in config["options"] else None,
                format_func=fmt,
            )
        else:
            return st.selectbox(
                label,
                config["options"],
                index=config["options"].index(value) if value in config["options"] else None
            )

    if config["widget"] == "date":
        return st.date_input(
            label, value=value if value is not None else None
            )

    if config["widget"] == "check":
        return st.checkbox(label, value=value if value is not None else None) # type: ignore

    raise ValueError(f"Unknown widget type for field {key}")
