"""CDSS configuration settings"""

# MongoDB settings
MONGODB_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "ckd_cdss"
COLLECTION_NAME = "patients"

# Model settings
MODEL_FOLDER_PATH = "models/ucickd/bnn_20251225_192634"

# Feature mapping from database fields to model features
FEATURE_MAP = {
    'bin__cad': 'cad',
    'num__hemo': 'hemo',
    'bin__htn': 'htn',
    'num__bp': 'bp',
    'bin__appet_poor': 'appet',
    'num__sg': 'sg',
    'num__pcv': 'pcv',
    'num__age': 'age',
    'bin__dm': 'dm'
}

# Clinical thresholds
UNCERTAINTY_THRESHOLDS = {
    'high_confidence': 0.05,
    'moderate_confidence': 0.10,
}

PROBABILITY_THRESHOLDS = {
    'very_high': 0.95,
    'high': 0.80,
    'borderline_low': 0.40,
    'borderline_high': 0.60,
    'low': 0.20,
    'very_low': 0.05,
}

# Image path
DIPSTICK_IMAGE_FOLDER = "src/cdss/dipstick_image"
