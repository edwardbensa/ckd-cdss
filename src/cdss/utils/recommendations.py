"""CDSS recommendation utils"""
# Imports
import streamlit as st
from src.cdss.config import UNCERTAINTY_THRESHOLDS, PROBABILITY_THRESHOLDS

def generate_recommendations(prob, uncertainty, shap_top_features=None):
    '''Generate clinical recommendations based on prediction and uncertainty'''

    recommendations = {
        "decision": "",
        "confidence": "",
        "action": "",
        "follow_up": "",
        "alert_level": ""
    }

    # High confidence CKD diagnosis
    if prob > PROBABILITY_THRESHOLDS['very_high'] and uncertainty < UNCERTAINTY_THRESHOLDS['high_confidence']:
        recommendations["decision"] = "CKD Diagnosis - High Confidence"
        recommendations["confidence"] = "Very High"
        recommendations["action"] = "Initiate CKD treatment protocols immediately"
        recommendations["follow_up"] = "Refer to nephrology within 1 week"
        recommendations["alert_level"] = "critical"

    # Moderate-high confidence CKD
    elif prob > PROBABILITY_THRESHOLDS['high'] and uncertainty < UNCERTAINTY_THRESHOLDS['moderate_confidence']:
        recommendations["decision"] = "CKD Diagnosis - Moderate Confidence"
        recommendations["confidence"] = "High"
        recommendations["action"] = "Initiate CKD treatment protocols"
        recommendations["follow_up"] = "Nephrology referral within 2 weeks, repeat labs in 1 month"
        recommendations["alert_level"] = "warning"

    # Uncertain CKD indication
    elif prob > 0.50 and (uncertainty > UNCERTAINTY_THRESHOLDS['moderate_confidence'] or 
                          PROBABILITY_THRESHOLDS['borderline_low'] < prob < PROBABILITY_THRESHOLDS['borderline_high']):
        recommendations["decision"] = "Possible CKD - Requires Clinical Review"
        recommendations["confidence"] = "Uncertain"
        recommendations["action"] = "Order additional tests: repeat creatinine, ACR, renal ultrasound"
        recommendations["follow_up"] = "Clinical assessment within 1 week, repeat testing in 2 weeks"
        recommendations["alert_level"] = "warning"

    # High confidence no CKD
    elif prob < PROBABILITY_THRESHOLDS['very_low'] and uncertainty < UNCERTAINTY_THRESHOLDS['high_confidence']:
        recommendations["decision"] = "No CKD - High Confidence"
        recommendations["confidence"] = "Very High"
        recommendations["action"] = "No immediate treatment needed"
        recommendations["follow_up"] = "Routine monitoring, annual screening"
        recommendations["alert_level"] = "success"

    # Moderate confidence no CKD
    elif prob < PROBABILITY_THRESHOLDS['low'] and uncertainty < UNCERTAINTY_THRESHOLDS['moderate_confidence']:
        recommendations["decision"] = "No CKD - Moderate Confidence"
        recommendations["confidence"] = "High"
        recommendations["action"] = "No immediate treatment needed"
        recommendations["follow_up"] = "Monitor risk factors, rescreen in 6 months"
        recommendations["alert_level"] = "success"

    # Uncertain no CKD
    else:
        recommendations["decision"] = "Uncertain - Clinical Judgment Required"
        recommendations["confidence"] = "Low"
        recommendations["action"] = "Comprehensive clinical assessment required"
        recommendations["follow_up"] = "Consider additional testing, specialist consultation"
        recommendations["alert_level"] = "info"

    # Add key contributing factors if available
    if shap_top_features:
        recommendations["key_factors"] = shap_top_features

    return recommendations



def alert_box(recommendations):
    """Render alerts for different recomendations types."""
    alert_type = recommendations["alert_level"]
    if alert_type == "critical":
        st.error(f"{recommendations['decision']}", icon=":material/error:")
    elif alert_type == "warning":
        st.warning(f"{recommendations['decision']}", icon=":material/warning:")
    elif alert_type == "success":
        st.success(f"{recommendations['decision']}", icon=":material/diagnosis:")
    else:
        st.info(f"{recommendations['decision']}", icon=":material/diagnosis:")
