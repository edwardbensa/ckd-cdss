# CKD Clinical Decision Support System (CDSS)

This repository contains a comprehensive **Clinical Decision Support System (CDSS)** for Chronic Kidney Disease (CKD). The system leverages **Bayesian Neural Networks (BNN)** to provide diagnostic predictions with built-in uncertainty quantification and features a computer vision component to extract clinical data from **urinalysis dipstick images**.

## Key Features

- **Bayesian Neural Network Diagnosis:** Predicts CKD status while providing a "confidence level" (uncertainty) to help clinicians identify borderline cases.
    
- **Computer Vision Integration:** Automated reading of urine dipstick images to populate clinical fields (Glucose, Specific Gravity, RBC, etc.).
    
- **Interpretability with SHAP:** Explains model predictions by highlighting the most influential patient features.
    
- **RAG-Powered Clinical Chat:** A retrieval-augmented generation agent that provides clinical guidance based on the **NICE NG203** guidelines.
    
- **Comprehensive Patient Management:** Full CRUD (Create, Read, Update, Delete) operations powered by MongoDB.
    
- **Analytics Dashboard:** System-wide insights into CKD prevalence, testing timelines, and demographic distributions.
    

---

## System Architecture

The application is built using a modular Python architecture:

|Component|Description|
|---|---|
|**Frontend**|Streamlit multi-page application (`app.py`).|
|**Database**|MongoDB for secure storage of patient records and diagnostic history (`db.py`).|
|**Model Logic**|Bayesian inference and preprocessing via `joblib` and `scikit-learn` (`models.py`).|
|**Vision**|Image processing utilities for dipstick analysis (`misc.py`).|
|**Guidelines**|RAG agent connected to a FastAPI backend for NICE guideline queries (`chat.py`).|

## Module Overview

### 1. Diagnostic Engine (`models.py`, `predictions.py`)

The system uses a pre-trained BNN. Unlike standard neural networks, the BNN provides an **uncertainty metric**.

- **Total Uncertainty:** Combines Aleatoric and Epistemic uncertainty.
    
- **Decision Logic:** Predictions are mapped to clinical recommendations (e.g., "Refer to nephrology within 1 week") based on both probability and confidence thresholds.
    

### 2. Dipstick Analysis (`data.py`, `patients.py`)

Clinicians can upload photos of urinalysis dipsticks. The system:

1. Processes the image to identify color pads.
    
2. Maps colors to clinical values (e.g., `neg`, `100(5.5)`, `500(28)`).
    
3. Automatically populates the patient form in the **Patient Management** tab.
    

### 3. Clinical Recommendations (`recommendations.py`)

Automated logic that generates actionable steps:

- **Critical:** High probability + High confidence.
    
- **Warning:** Moderate probability or high uncertainty (requires clinical review).
    
- **Success:** High confidence "No CKD" result.
    

### 4. Analytics & Reporting (`analytics.py`)

Visualizes data using `Plotly`:

- Uncertainty vs. Probability scatter plots to identify "edge cases."
    
- Distribution of CKD cases across different age groups and comorbidities (HTN, DM).
    
- Cumulative testing timelines.
    

---

## Installation & Setup

### Prerequisites

- Python 3.9+
    
- MongoDB Instance (Local or Atlas)
    
- RAG API (FastAPI) running at `http://localhost:8000` (for the Chat feature)
    

### Steps

1. **Clone the repository:**
```bash
git clone <repository-url>
cd ckd-cdss
```
2. **Install dependencies:**
```bash
pip install -r requirements.txt
```
3. **Configure Environment:** Update `src/cdss/config.py` with your `MONGODB_URI` and `MODEL_FOLDER_PATH`.
4. Run the Application:
```bash
streamlit run app.py
```

## Clinical Workflow

1. **Data Entry:** Add a patient via "Patient Management." Use the **Dipstick Image Upload** to save time on manual entry.
    
2. **Diagnostic Testing:** Go to "Diagnostic Prediction," select the patient, and run the model.
    
3. **Review:** Examine the probability score and uncertainty. Check the SHAP values (if enabled) to understand _why_ the model made that choice.
    
4. **Action:** Follow the generated clinical recommendation and save the results to the patient's permanent record.
    
5. **Consult:** Use the "Guideline Chat" to ask specific questions about NICE protocols (NG203) regarding the patient's specific presentation.
    

---

## Disclaimer

_This system is intended for research and as a clinical decision support tool. It is not a replacement for professional clinical judgment._