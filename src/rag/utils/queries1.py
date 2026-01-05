"""Professional Queries for testing RAG agent"""

# Diagnosis and Classification
T1 = {
    1: "How do I diagnose CKD and confirm its severity?",
    2: "How do I classify CKD using GFR and ACR, and determine the patient's risk category?",
    3: "3.	What investigations should I order for suspected CKD and why?",
    4: "4.	What does ACR mean, and why is it important in CKD?",
    5: "5.	Why do we check for protein in the urine, and what does it indicate?"
}

# Monitoring and Progression
T2 = {
    1: "How do I monitor CKD progression over time?",
    2: "How frequently should I repeat eGFR and ACR in stable CKD?",
    3: "What counts as a significant drop in eGFR?",
    4: "How do I identify rapid CKD progression?",
    5: "How do I interpret fluctuating eGFR results?"
}

# Referral and Risk Assessment
T3 = {
    1: "When should I refer a CKD patient to nephrology?",
    2: "What should I do if ACR is high but eGFR is normal?",
    3: "How do I use the GFR/ACR risk grid to assess CKD risk?",
}

# Management
T4 = {
    1: "How do I manage CKD in someone with diabetes?",
    2: "How do I manage CKD in pregnancy?",
    3: "What lifestyle advice should I give to someone with CKD?",
    4: "How should I manage blood pressure in a CKD patient with diabetes and albuminuria?",
    5: "Why do we avoid NSAIDs in CKD?",
    6: "What's the rationale for offering statins to CKD patients?"
}

# Misc and Special Scenarios
T5 = {
    1: "How do I assess and manage anaemia in CKD, including iron therapy?",
    2: "How do I optimise erythropoiesis in CKD patients?",
    3: "When is renal ultrasound indicated, and what's the rationale behind it?",
    4: "My patient has CKD stage 3b with rising ACR. What should I do next?",
    5: "What monitoring schedule should I use for an older adult with CKD stage 4?",
    6: "My patient's eGFR dropped from 52 to 41 in a year. Is this rapid progression?"
}

QUERIES = [T1, T2, T3, T4, T5]
