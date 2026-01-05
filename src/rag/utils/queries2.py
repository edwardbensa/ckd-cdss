"""Queries for testing RAG agent"""

# General CKD management
T1 = {
    1: "When should I refer a CKD patient to nephrology?",
    2: "How often should I check ACR in someone with stable CKD?",
    3: "What blood pressure target should I aim for in CKD?",
    4: "How do I manage CKD in someone with diabetes?",
    5: "What lifestyle advice should I give to someone with CKD?"
}

# Monitoring & progression
T2 = {
    1: "How do I monitor CKD progression over time?",
    2: "Which tests indicate worsening kidney function?",
    3: "How frequently should I repeat eGFR and ACR?",
    4: "What counts as a significant drop in eGFR?",
    5: "How do I identify rapid CKD progression?"
}

# Rationales
T3 = {
    1: "Why do we check ACR in CKD?",
    2: "What's the reasoning behind monitoring potassium in CKD?",
    3: "Why is blood pressure control important in CKD?",
    4: "What's the rationale for offering statins to CKD patients?",
    5: "Why do we avoid NSAIDs in CKD?"
}

# Table‑focused queries
T4 = {
    1: "Show me the GFR/ACR risk grid.",
    2: "How do I classify CKD using GFR and ACR?",
    3: "What are the CKD risk categories?",
    4: "What monitoring frequency should I use for CKD stages?",
    5: "What's the high-dose IV iron regimen table?"
}

# Ambiguous / colloquial queries
T5 = {
    1: "How bad is stage 3 CKD?",
    2: "What does ACR actually mean?",
    3: "What kidney tests should I repeat regularly?",
    4: "When should I worry about declining kidney function?",
    5: "What's the deal with protein in the urine?"
}

# Edge‑case clinical reasoning
T6 = {
    1: "How do I manage CKD in pregnancy?",
    2: "What should I do if ACR is high but eGFR is normal?",
    3: "How do I interpret fluctuating eGFR results?",
    4: "What if ACR suddenly doubles?",
    5: "How do I assess CKD in older adults?"
}

# Multi‑chunk retrieval stress tests
T7 = {
    1: "How do I diagnose CKD and classify its severity?",
    2: "What investigations should I order for suspected CKD?",
    3: "How do I manage anaemia in CKD?",
    4: "What are the indications for renal ultrasound?",
    5: "How do I optimise erythropoiesis in CKD patients?"
}

# Adversarial / Hallucination‑Resistance Queries
T8 = {
    1: "Does NG203 recommend using herbal supplements to slow CKD progression?",
    2: "What does NICE say about reversing CKD completely?",
    3: "Is there a cure for CKD mentioned in the guideline?",
    4: "Does NG203 recommend a special CKD diet like keto or paleo?",
    5: "What does NG203 say about stem cell therapy for CKD?"
}

# Patient‑Specific Queries (for testing patient‑mode)
T9 = {
    1: "My patient has CKD stage 3b with rising ACR. What should I do next?",
    2: "How should I manage blood pressure in a CKD patient with diabetes and albuminuria?",
    3: "My patient's eGFR dropped from 52 to 41 in a year. Is this rapid progression?",
    4: "What monitoring schedule should I use for an older adult with CKD stage 4?",
    5: "My patient has anaemia and CKD. How do I assess iron status and decide on treatment?"
}

# Multi‑Source Retrieval Stress Tests
T10 = {
    1: "How do I classify CKD and determine the patient's risk category?",
    2: "What investigations should I order for suspected CKD and why?",
    3: "How should I monitor CKD progression and what evidence supports this?",
    4: "How do I assess and manage anaemia in CKD, including iron therapy?",
    5: "When is renal ultrasound indicated and what's the rationale behind it?"
}

QUERIES = [T1, T2, T3, T4, T5, T6, T7, T8, T9, T10]
