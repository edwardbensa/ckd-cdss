"""Dataset utility functions for tabular data processing and model training."""

# Imports
import numpy as np
import pandas as pd

def compute_bmi(height, weight):
    """Calculate BMI using height and weight."""
    bmi = weight / (height / 100) ** 2
    return round(bmi, 1)

def simplify_ethnicity(ethnicity):
    """Simplify the ethnicity entry."""
    s_eth = ethnicity.apply(
        lambda x: "Mixed" if "Mixed" in x else
        "White" if "White" in x else
        "Black" if "Black" in x else
        "Asian" if "Asian" in x else
        "South Asian" if "Indian" in x or "Pakistani" in x or "Bangladeshi" in x else
        "Other")
    return s_eth

def black_ancestry(ethnicity):
    """Use ethnicity to determine if there is black ancestry."""
    eth_black = ethnicity.apply(lambda x: True if "Black" in x else False)
    return eth_black


def impute_acr(row, seed=42):
    """Impute ACR based on confirmed_albuminuria flag."""
    acr = 0
    if row["confirmed_albuminuria"] is False and pd.isna(row["acr"]):
        # Impute based on normal distribution
        rng = np.random.default_rng(seed)
        acr = rng.choice([0.0, 1.0, 2.0], p=[0.274, 0.452, 0.274])
    else:
        acr = row["acr"]
    return acr


def gfr_staging(row):
    """Determine GFR stage based on egfr and acr values."""
    stage = ""
    if row["egfr"] >= 90 and row["acr"] < 3:
        stage = "notckd"
    elif row["egfr"] >= 90:
        stage = "G1"
    elif 89 >= row["egfr"] >= 60:
        stage = "G2"
    elif 59 >= row["egfr"] >= 45:
        stage = "G3a"
    elif 44 >= row["egfr"] >= 30:
        stage = "G3b"
    elif 29 >= row["egfr"] >= 15:
        stage = "G4"
    elif row["egfr"] < 15:
        stage = "G5"

    return stage

def acr_staging(row):
    """Determine ACR stage based on acr values."""
    stage = ""
    if row["acr"] < 3 and row["egfr"] >= 90:
        stage = "notckd"
    elif 0 <= row["acr"] < 3:
        stage = "A1"
    elif 30 >= row["acr"] >= 3:
        stage = "A2"
    elif row["acr"] > 30:
        stage = "A3"

    return stage

def ckd_status(row):
    """Determine CKD status using gfr stage."""
    return "ckd" if row["gfr_stage"] != "notckd" else "notckd"
