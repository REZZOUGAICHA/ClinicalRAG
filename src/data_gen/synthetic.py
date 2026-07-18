"""
Generates synthetic clinical PDF reports for development and testing.

All data is entirely fictional. No real patient information is used.
PHI-redaction note: even with synthetic data we structure reports the same way
real reports are structured, so the PHI-redaction pass we add in Week 2 can be
tested against realistic field placements.
"""

import os
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT


REPORTS = [
    {
        "filename": "report_001_lymphoma_staging.pdf",
        "patient_id": "SYN-2024-001",
        "date": "2024-03-15",
        "report_type": "Oncology Consultation Report",
        "sections": {
            "PATIENT INFORMATION": (
                "Patient ID: SYN-2024-001\n"
                "Date of Birth: 1978-06-22  |  Sex: Female  |  Age: 45\n"
                "Referring Physician: Dr. M. Benali\n"
                "Report Date: 2024-03-15"
            ),
            "CHIEF COMPLAINT": (
                "Patient presents with painless cervical lymphadenopathy persisting for "
                "6 weeks, associated with night sweats and unintentional weight loss of 5 kg."
            ),
            "DIAGNOSIS": (
                "Hodgkin Lymphoma, Nodular Sclerosis subtype (WHO Classification).\n"
                "Ann Arbor Stage: IIB\n"
                "Involvement: Bilateral cervical and mediastinal lymph nodes.\n"
                "B-symptoms: Present (fever, night sweats, weight loss >10% body weight)."
            ),
            "LABORATORY RESULTS": (
                "Complete Blood Count:\n"
                "  WBC: 11.2 × 10⁹/L  (Reference: 4.0–11.0)  [HIGH]\n"
                "  Hemoglobin: 10.8 g/dL  (Reference: 12.0–16.0)  [LOW]\n"
                "  Platelets: 387 × 10⁹/L  (Reference: 150–400)  [Normal]\n"
                "  Lymphocytes: 0.9 × 10⁹/L  (Reference: 1.0–4.8)  [LOW]\n\n"
                "Inflammatory Markers:\n"
                "  ESR: 78 mm/h  (Reference: <20)  [HIGH]\n"
                "  CRP: 42 mg/L  (Reference: <5)  [HIGH]\n"
                "  LDH: 312 U/L  (Reference: 140–280)  [HIGH]\n\n"
                "Liver Function:\n"
                "  ALT: 28 U/L  |  AST: 31 U/L  |  Bilirubin: 0.9 mg/dL  — All Normal"
            ),
            "IMAGING": (
                "PET-CT (2024-03-10): Hypermetabolic bilateral cervical lymph nodes, largest "
                "2.8 cm in diameter, SUVmax 8.4. Mediastinal involvement confirmed. "
                "No infradiaphragmatic disease. No bone marrow involvement.\n\n"
                "Chest X-ray: Widened mediastinum consistent with lymphadenopathy."
            ),
            "MEDICATIONS": (
                "Current medications:\n"
                "  - Prednisolone 40 mg/day PO (bridging, to be tapered)\n"
                "  - Paracetamol 1g PRN for fever\n"
                "  - No known drug allergies"
            ),
            "TREATMENT PLAN": (
                "Initiate ABVD chemotherapy protocol (Adriamycin, Bleomycin, Vinblastine, "
                "Dacarbazine). Planned 4 cycles with interim PET-CT after cycle 2 to assess "
                "response. Referral to radiation oncology for consolidation radiotherapy "
                "evaluation. Fertility preservation counseling offered and accepted."
            ),
            "FOLLOW-UP": (
                "Next appointment: 2024-03-29 (1 week prior to cycle 1).\n"
                "Patient education provided. Emergency contact for febrile neutropenia given."
            ),
        },
    },
    {
        "filename": "report_002_diabetes_management.pdf",
        "patient_id": "SYN-2024-002",
        "date": "2024-04-02",
        "report_type": "Endocrinology Follow-Up Report",
        "sections": {
            "PATIENT INFORMATION": (
                "Patient ID: SYN-2024-002\n"
                "Date of Birth: 1965-11-08  |  Sex: Male  |  Age: 58\n"
                "Referring Physician: Dr. A. Meziane\n"
                "Report Date: 2024-04-02"
            ),
            "CHIEF COMPLAINT": (
                "Routine diabetes follow-up. Patient reports increased fatigue and polyuria "
                "over the past 4 weeks. Home glucose readings consistently above 200 mg/dL."
            ),
            "DIAGNOSIS": (
                "Type 2 Diabetes Mellitus — Inadequately controlled (ICD-10: E11.65).\n"
                "Comorbidities: Hypertension, Dyslipidemia, Mild chronic kidney disease (Stage 2)."
            ),
            "LABORATORY RESULTS": (
                "Glycemic Control:\n"
                "  HbA1c: 9.2%  (Target: <7.0%)  [ABOVE TARGET]\n"
                "  Fasting Glucose: 214 mg/dL  (Reference: 70–100)  [HIGH]\n"
                "  Post-prandial Glucose (2h): 287 mg/dL  (Reference: <140)  [HIGH]\n\n"
                "Renal Function:\n"
                "  Creatinine: 1.3 mg/dL  |  eGFR: 62 mL/min/1.73m²  (Stage 2 CKD)\n"
                "  Urine Albumin-to-Creatinine Ratio: 48 mg/g  (Reference: <30)  [HIGH]\n\n"
                "Lipid Panel:\n"
                "  Total Cholesterol: 218 mg/dL  |  LDL: 142 mg/dL  [HIGH]\n"
                "  HDL: 38 mg/dL  [LOW]  |  Triglycerides: 198 mg/dL  [HIGH]"
            ),
            "IMAGING": (
                "Renal ultrasound (2024-03-20): Bilateral kidneys normal in size and echogenicity. "
                "No hydronephrosis. Findings consistent with early diabetic nephropathy."
            ),
            "MEDICATIONS": (
                "Current medications:\n"
                "  - Metformin 1000 mg BID PO\n"
                "  - Sitagliptin 100 mg OD PO\n"
                "  - Amlodipine 5 mg OD PO (hypertension)\n"
                "  - Atorvastatin 20 mg OD PO (dyslipidemia)\n"
                "  MODIFICATION: Add Empagliflozin 10 mg OD (cardiorenal benefit, eGFR permits)"
            ),
            "TREATMENT PLAN": (
                "1. Intensify glycemic control: add Empagliflozin 10 mg OD.\n"
                "2. Increase Atorvastatin to 40 mg OD — LDL target <100 mg/dL.\n"
                "3. Low-sodium, low-carbohydrate diet reinforcement. Dietitian referral placed.\n"
                "4. Blood pressure target: <130/80 mmHg. Consider ACE inhibitor for renal protection.\n"
                "5. Ophthalmology referral for annual diabetic retinopathy screening."
            ),
            "FOLLOW-UP": (
                "HbA1c recheck in 3 months (2024-07-02).\n"
                "Renal function panel in 6 weeks to assess Empagliflozin response.\n"
                "Patient instructed to monitor fasting glucose daily and log readings."
            ),
        },
    },
    {
        "filename": "report_003_acute_mi.pdf",
        "patient_id": "SYN-2024-003",
        "date": "2024-02-18",
        "report_type": "Cardiology Discharge Summary",
        "sections": {
            "PATIENT INFORMATION": (
                "Patient ID: SYN-2024-003\n"
                "Date of Birth: 1955-03-14  |  Sex: Male  |  Age: 69\n"
                "Admission Date: 2024-02-15  |  Discharge Date: 2024-02-18\n"
                "Attending Cardiologist: Dr. K. Hadj"
            ),
            "CHIEF COMPLAINT": (
                "Emergency admission with acute onset crushing substernal chest pain radiating "
                "to left arm, diaphoresis and nausea. Duration 45 minutes before arrival. "
                "ECG on admission: ST-elevation in leads II, III, aVF."
            ),
            "DIAGNOSIS": (
                "Acute Inferior ST-Elevation Myocardial Infarction (STEMI).\n"
                "Culprit vessel: Right Coronary Artery (RCA) — 95% occlusion.\n"
                "TIMI flow post-PCI: 3 (complete reperfusion achieved).\n"
                "Left Ventricular Ejection Fraction (post-procedure): 48%."
            ),
            "LABORATORY RESULTS": (
                "Cardiac Biomarkers (on admission):\n"
                "  Troponin I (0h): 2.8 ng/mL  (Reference: <0.04)  [CRITICALLY HIGH]\n"
                "  Troponin I (6h): 18.4 ng/mL  [CRITICALLY HIGH — peak]\n"
                "  CK-MB: 142 U/L  (Reference: <25)  [HIGH]\n"
                "  BNP: 310 pg/mL  (Reference: <100)  [HIGH]\n\n"
                "Complete Blood Count:\n"
                "  WBC: 13.1 × 10⁹/L  [HIGH — stress response]\n"
                "  Hemoglobin: 13.8 g/dL  |  Platelets: 224 × 10⁹/L  — Normal\n\n"
                "Metabolic Panel:\n"
                "  Creatinine: 1.1 mg/dL  |  eGFR: 74  |  K⁺: 4.1 mEq/L — Normal\n"
                "  Glucose on admission: 186 mg/dL  [HIGH — stress hyperglycemia]"
            ),
            "IMAGING": (
                "ECG on admission: ST-elevation leads II, III, aVF. Reciprocal changes in I, aVL.\n\n"
                "Echocardiogram (2024-02-16): Inferior wall hypokinesia. EF 48%. "
                "Mild mitral regurgitation. No pericardial effusion.\n\n"
                "Coronary angiography: RCA 95% proximal occlusion. LAD and LCx patent. "
                "Successful PCI with drug-eluting stent placement."
            ),
            "MEDICATIONS": (
                "Discharge medications:\n"
                "  - Aspirin 100 mg OD PO (lifelong)\n"
                "  - Ticagrelor 90 mg BID PO (dual antiplatelet, 12 months)\n"
                "  - Atorvastatin 80 mg OD PO (high-intensity statin)\n"
                "  - Ramipril 5 mg OD PO (ACE inhibitor, cardiac remodeling)\n"
                "  - Bisoprolol 5 mg OD PO (beta-blocker, rate control)\n"
                "  - Pantoprazole 40 mg OD PO (GI protection with dual antiplatelet)"
            ),
            "TREATMENT PLAN": (
                "Post-MI rehabilitation program enrollment arranged.\n"
                "Lifestyle modification counseling: smoking cessation (patient is a smoker — "
                "12 pack-years), Mediterranean diet, graduated exercise.\n"
                "Follow-up echocardiogram in 6 weeks to reassess EF.\n"
                "Strict avoidance of NSAIDs while on dual antiplatelet therapy."
            ),
            "FOLLOW-UP": (
                "Cardiology clinic: 2024-03-04 (2 weeks post-discharge).\n"
                "GP follow-up: 1 week for wound check and BP monitoring.\n"
                "Patient advised not to drive for 4 weeks per local guidelines."
            ),
        },
    },
    {
        "filename": "report_004_pneumonia.pdf",
        "patient_id": "SYN-2024-004",
        "date": "2024-05-10",
        "report_type": "Respiratory Medicine Report",
        "sections": {
            "PATIENT INFORMATION": (
                "Patient ID: SYN-2024-004\n"
                "Date of Birth: 1990-09-30  |  Sex: Female  |  Age: 33\n"
                "Referring Physician: Dr. S. Ait-Ali\n"
                "Report Date: 2024-05-10"
            ),
            "CHIEF COMPLAINT": (
                "Progressive productive cough with purulent sputum, fever (38.9°C), "
                "right-sided pleuritic chest pain and dyspnea over 5 days."
            ),
            "DIAGNOSIS": (
                "Community-Acquired Pneumonia (CAP), right lower lobe.\n"
                "Severity: CURB-65 score 1 — Mild. Managed as outpatient.\n"
                "Likely bacterial etiology (Streptococcus pneumoniae most probable)."
            ),
            "LABORATORY RESULTS": (
                "Infection Markers:\n"
                "  WBC: 16.4 × 10⁹/L  (Reference: 4–11)  [HIGH]\n"
                "  Neutrophils: 13.2 × 10⁹/L  [HIGH — consistent with bacterial infection]\n"
                "  CRP: 98 mg/L  (Reference: <5)  [HIGH]\n"
                "  Procalcitonin: 0.8 ng/mL  (Reference: <0.1)  [MODERATE ELEVATION]\n\n"
                "Sputum Culture: Pending at time of report. Blood cultures: No growth at 48h.\n\n"
                "Arterial Blood Gas (room air):\n"
                "  PaO2: 82 mmHg  |  SaO2: 96%  |  pH: 7.42  — Mild hypoxemia"
            ),
            "IMAGING": (
                "Chest X-ray (PA, 2024-05-10): Right lower lobe consolidation with air bronchograms. "
                "No pleural effusion. No pneumothorax. Cardiac silhouette normal.\n\n"
                "CT Chest (not performed — X-ray sufficient for clinical diagnosis and management)."
            ),
            "MEDICATIONS": (
                "Prescribed:\n"
                "  - Amoxicillin-Clavulanate 1g/125mg TID PO × 7 days\n"
                "  - Azithromycin 500 mg OD PO × 5 days (atypical coverage)\n"
                "  - Paracetamol 1g QID PRN (fever/pain)\n"
                "  - Adequate hydration encouraged\n"
                "  No known drug allergies."
            ),
            "TREATMENT PLAN": (
                "Outpatient antibiotic therapy as above. Patient instructed on:\n"
                "  - Return to ED if: SpO2 <94%, confusion, worsening dyspnea, or no improvement by day 3.\n"
                "  - Complete full antibiotic course.\n"
                "  - Rest, increased fluid intake.\n"
                "Pneumococcal and influenza vaccination recommended at follow-up visit."
            ),
            "FOLLOW-UP": (
                "Review in 48–72 hours by telephone. Clinic review in 6 weeks with repeat chest X-ray "
                "to confirm radiological resolution (important to exclude underlying malignancy)."
            ),
        },
    },
    {
        "filename": "report_005_renal_failure.pdf",
        "patient_id": "SYN-2024-005",
        "date": "2024-06-05",
        "report_type": "Nephrology Consultation Report",
        "sections": {
            "PATIENT INFORMATION": (
                "Patient ID: SYN-2024-005\n"
                "Date of Birth: 1948-04-17  |  Sex: Male  |  Age: 76\n"
                "Referring Physician: Dr. F. Boukhari\n"
                "Report Date: 2024-06-05"
            ),
            "CHIEF COMPLAINT": (
                "Referred for evaluation of progressively worsening renal function. "
                "Creatinine trend: 1.4 mg/dL (Jan 2024) → 2.1 mg/dL (Mar 2024) → 3.2 mg/dL (Jun 2024). "
                "Associated ankle oedema and reduced urine output."
            ),
            "DIAGNOSIS": (
                "Chronic Kidney Disease, Stage 4 (eGFR 18 mL/min/1.73m²).\n"
                "Etiology: Hypertensive nephrosclerosis and diabetic nephropathy.\n"
                "Complications: Anaemia of CKD, Secondary hyperparathyroidism, Metabolic acidosis."
            ),
            "LABORATORY RESULTS": (
                "Renal Function:\n"
                "  Creatinine: 3.2 mg/dL  |  eGFR: 18 mL/min/1.73m²  (Stage 4 CKD)\n"
                "  Urea: 68 mg/dL  (Reference: 10–50)  [HIGH]\n"
                "  Uric Acid: 8.4 mg/dL  [HIGH]\n\n"
                "Electrolytes:\n"
                "  Potassium: 5.6 mEq/L  [HIGH — risk of arrhythmia]\n"
                "  Bicarbonate: 17 mEq/L  (Reference: 22–29)  [LOW — metabolic acidosis]\n"
                "  Phosphate: 5.8 mg/dL  [HIGH]\n"
                "  Calcium (corrected): 8.1 mg/dL  [LOW-NORMAL]\n\n"
                "Anaemia Workup:\n"
                "  Hemoglobin: 9.4 g/dL  [LOW]  |  Ferritin: 48 ng/mL  [LOW]\n"
                "  EPO level: Inappropriately low for degree of anaemia\n\n"
                "PTH: 312 pg/mL  (Reference: 15–65)  [HIGH — secondary hyperparathyroidism]"
            ),
            "IMAGING": (
                "Renal ultrasound: Bilateral shrunken kidneys (right 8.2 cm, left 8.0 cm). "
                "Increased echogenicity. Corticomedullary differentiation reduced. "
                "No obstruction. Findings consistent with chronic parenchymal disease."
            ),
            "MEDICATIONS": (
                "Initiated / Modified:\n"
                "  - Sodium bicarbonate 500 mg TID PO (correct metabolic acidosis)\n"
                "  - Sevelamer 800 mg TID with meals (phosphate binder)\n"
                "  - Alfacalcidol 0.25 mcg OD PO (active Vitamin D)\n"
                "  - Iron sucrose IV 200 mg (correct iron deficiency)\n"
                "  - Erythropoietin-stimulating agent (darbepoetin alfa 60 mcg SC q2 weeks)\n"
                "  - Potassium-restricted diet — dietitian referral placed\n"
                "  STOPPED: NSAIDs, metformin (contraindicated at this eGFR)"
            ),
            "TREATMENT PLAN": (
                "Renal Replacement Therapy planning initiated. Patient counselled on:\n"
                "  1. Haemodialysis (HD) via arteriovenous fistula — surgical referral placed\n"
                "  2. Peritoneal dialysis (PD) — patient to consider home-based option\n"
                "  3. Pre-emptive transplant evaluation — referred to transplant centre\n"
                "Target Hb: 10–12 g/dL. Monitor potassium weekly until stable."
            ),
            "FOLLOW-UP": (
                "Nephrology clinic: 2024-06-19 (2 weeks) — renal function panel, electrolytes.\n"
                "Vascular surgery consultation for AV fistula creation: 2024-06-12.\n"
                "Transplant workup appointment: pending."
            ),
        },
    },
    {
        "filename": "report_006_stroke.pdf",
        "patient_id": "SYN-2024-006",
        "date": "2024-01-22",
        "report_type": "Neurology Discharge Summary",
        "sections": {
            "PATIENT INFORMATION": (
                "Patient ID: SYN-2024-006\n"
                "Date of Birth: 1960-07-03  |  Sex: Female  |  Age: 63\n"
                "Admission Date: 2024-01-20  |  Discharge Date: 2024-01-22\n"
                "Attending Neurologist: Dr. L. Kaci"
            ),
            "CHIEF COMPLAINT": (
                "Sudden onset right-sided facial droop, right arm weakness (3/5 power) "
                "and expressive dysphasia. Onset witnessed at 09:15. Patient arrived at ED at 09:55. "
                "NIHSS score on admission: 11."
            ),
            "DIAGNOSIS": (
                "Acute Ischaemic Stroke, left MCA territory (M2 branch).\n"
                "Etiology: Cardioembolic (Atrial Fibrillation — newly diagnosed).\n"
                "IV thrombolysis administered (tPA 0.9 mg/kg) at 10:22 (door-to-needle: 27 min).\n"
                "Post-thrombolysis NIHSS at 24h: 4 (significant improvement)."
            ),
            "LABORATORY RESULTS": (
                "Coagulation:\n"
                "  INR: 1.1  |  APTT: 28s  |  Platelets: 198 × 10⁹/L  — Normal (pre-thrombolysis)\n\n"
                "Metabolic:\n"
                "  Glucose on admission: 7.8 mmol/L  |  HbA1c: 6.1%  (pre-diabetic)\n"
                "  Lipid panel: LDL 3.8 mmol/L  [HIGH]  |  Total Cholesterol: 5.9 mmol/L\n\n"
                "Cardiac:\n"
                "  Troponin: 0.08 ng/mL  [mildly elevated — demand ischaemia]\n"
                "  BNP: 180 pg/mL  [elevated — AF with reduced function]"
            ),
            "IMAGING": (
                "CT Brain (non-contrast, on admission): No haemorrhage. No established infarct. "
                "ASPECTS score: 9/10. Thrombolysis safe to proceed.\n\n"
                "MRI Brain (DWI, 2024-01-21): Left MCA M2 territory infarct confirmed. "
                "No haemorrhagic transformation. Infarct volume ~8 mL.\n\n"
                "CT Angiography: M2 branch occlusion. No large vessel occlusion suitable for thrombectomy. "
                "No significant carotid stenosis."
            ),
            "MEDICATIONS": (
                "Discharge medications:\n"
                "  - Apixaban 5 mg BID PO (anticoagulation for AF — started day 3 post-stroke)\n"
                "  - Atorvastatin 80 mg OD PO\n"
                "  - Ramipril 5 mg OD PO (secondary prevention, BP management)\n"
                "  - Aspirin 100 mg OD PO (bridging until anticoagulation therapeutic)\n"
                "  STOPPED: Aspirin to be discontinued after 3 months per guideline"
            ),
            "TREATMENT PLAN": (
                "Inpatient physiotherapy and speech-language therapy initiated. "
                "Discharge to stroke rehabilitation unit. AF rate control: Bisoprolol 2.5 mg OD added. "
                "Echocardiogram arranged to assess LA thrombus and cardiac function. "
                "Driving prohibited for minimum 1 month (regulatory requirement)."
            ),
            "FOLLOW-UP": (
                "Stroke clinic: 2024-02-05 (2 weeks). Cardiology: 2024-01-30 (echo results).\n"
                "Community physiotherapy and speech therapy referrals placed.\n"
                "Patient and family education on AF management and stroke recurrence signs."
            ),
        },
    },
    {
        "filename": "report_007_depression_anxiety.pdf",
        "patient_id": "SYN-2024-007",
        "date": "2024-04-25",
        "report_type": "Psychiatry Outpatient Assessment",
        "sections": {
            "PATIENT INFORMATION": (
                "Patient ID: SYN-2024-007\n"
                "Date of Birth: 1995-12-14  |  Sex: Female  |  Age: 28\n"
                "Referring Physician: Dr. N. Saadi (GP)\n"
                "Report Date: 2024-04-25"
            ),
            "CHIEF COMPLAINT": (
                "Referred by GP for persistent low mood, anhedonia and anxiety over 4 months. "
                "Patient reports inability to attend work for 3 weeks, disrupted sleep, "
                "reduced appetite and passive suicidal ideation (no active plan or intent)."
            ),
            "DIAGNOSIS": (
                "Major Depressive Disorder, moderate severity (DSM-5).\n"
                "Comorbid: Generalised Anxiety Disorder (GAD).\n"
                "PHQ-9 score: 17 (moderate-to-severe depression).\n"
                "GAD-7 score: 14 (moderate anxiety).\n"
                "Suicide risk assessment: LOW — passive ideation, no plan, strong protective factors."
            ),
            "LABORATORY RESULTS": (
                "Screening to exclude organic causes:\n"
                "  TSH: 2.1 mIU/L  (Reference: 0.4–4.0)  — Normal (thyroid excluded)\n"
                "  Vitamin D: 18 ng/mL  (Reference: 30–100)  [DEFICIENT]\n"
                "  B12: 310 pg/mL  (Reference: 200–900)  — Normal\n"
                "  Full Blood Count: Normal\n"
                "  Fasting Glucose: 88 mg/dL  — Normal\n"
                "  Urine toxicology screen: Negative"
            ),
            "IMAGING": (
                "No neuroimaging indicated at this time. Brain MRI to be considered if treatment "
                "refractory or if atypical features emerge."
            ),
            "MEDICATIONS": (
                "Initiated:\n"
                "  - Sertraline 50 mg OD PO (increase to 100 mg at week 4 if tolerated)\n"
                "  - Vitamin D3 4000 IU OD PO × 8 weeks, then maintenance 1000 IU OD\n"
                "  Discussed: Benzodiazepines deferred — risk of dependence; sleep hygiene first.\n"
                "  No contraindications. No current medications. No known allergies."
            ),
            "TREATMENT PLAN": (
                "Combined pharmacological and psychological approach:\n"
                "  1. CBT referral — waitlist placed (estimated 6–8 weeks).\n"
                "  2. Sertraline titration as above — side effect counselling given.\n"
                "  3. Safety plan discussed and documented. Emergency contacts provided.\n"
                "  4. Sick note provided for 4 additional weeks. Gradual return-to-work plan.\n"
                "  5. Lifestyle: regular sleep schedule, daily light exercise, reduced caffeine."
            ),
            "FOLLOW-UP": (
                "Psychiatry review: 2024-05-09 (2 weeks) — medication response, tolerability, safety.\n"
                "If worsening, active suicidal ideation, or psychotic features emerge: "
                "attend ED immediately or call crisis line. Safety plan given to patient and carer."
            ),
        },
    },
    {
        "filename": "report_008_fracture_ortho.pdf",
        "patient_id": "SYN-2024-008",
        "date": "2024-03-30",
        "report_type": "Orthopaedic Surgery Report",
        "sections": {
            "PATIENT INFORMATION": (
                "Patient ID: SYN-2024-008\n"
                "Date of Birth: 1942-02-28  |  Sex: Female  |  Age: 82\n"
                "Admission Date: 2024-03-28  |  Procedure Date: 2024-03-29\n"
                "Operating Surgeon: Dr. Y. Taleb"
            ),
            "CHIEF COMPLAINT": (
                "Brought to ED by ambulance after mechanical fall at home. "
                "Unable to weight-bear on right hip. Significant pain on movement. "
                "Right leg externally rotated and shortened on examination."
            ),
            "DIAGNOSIS": (
                "Right intracapsular neck of femur fracture (Garden IV — displaced).\n"
                "Mechanism: Low-energy mechanical fall from standing height.\n"
                "Comorbidities relevant to surgical risk: Hypertension, Osteoporosis (T-score -3.1 lumbar spine)."
            ),
            "LABORATORY RESULTS": (
                "Pre-operative bloods:\n"
                "  Hemoglobin: 11.2 g/dL  [LOW — pre-op anaemia]\n"
                "  Platelets: 187 × 10⁹/L  |  INR: 1.0  — Normal coagulation\n"
                "  Creatinine: 1.1 mg/dL  |  eGFR: 58  (mild-moderate CKD)\n"
                "  Sodium: 138 mEq/L  |  Potassium: 4.0 mEq/L  — Normal\n"
                "  Albumin: 31 g/L  (Reference: 35–50)  [LOW — nutritional risk]\n\n"
                "Cardiac clearance: ECG — Normal sinus rhythm. No ischaemic changes. "
                "Echo not required (ASA Grade III, anaesthesia cleared)."
            ),
            "IMAGING": (
                "X-ray Hip AP and lateral (2024-03-28): Complete displaced intracapsular fracture "
                "right neck of femur. No periprosthetic fracture. No significant osteoarthritis.\n\n"
                "Post-operative X-ray (2024-03-29): Total hip replacement right side — "
                "well-positioned prosthesis, no dislocation, cement mantle satisfactory."
            ),
            "MEDICATIONS": (
                "Post-operative:\n"
                "  - Enoxaparin 40 mg SC OD (VTE prophylaxis × 35 days)\n"
                "  - Paracetamol 1g QID PO (regular analgesia)\n"
                "  - Tramadol 50 mg TID PRN PO (breakthrough pain — use cautiously in elderly)\n"
                "  - Omeprazole 20 mg OD PO (gastric protection)\n"
                "  - Alendronate 70 mg weekly PO (start 6 weeks post-op — osteoporosis)\n"
                "  - Calcium 500 mg + Vitamin D3 400 IU BID PO"
            ),
            "TREATMENT PLAN": (
                "Day 1 post-op mobilisation with physiotherapy. Weight-bearing as tolerated. "
                "Occupational therapy assessment for home modifications. "
                "Falls prevention programme referral. Nutritional supplementation started. "
                "Bone health clinic referral for osteoporosis management."
            ),
            "FOLLOW-UP": (
                "Orthopaedic clinic: 2024-04-26 (4 weeks) — wound check, X-ray, mobility assessment.\n"
                "GP: Enoxaparin prescription completion (35 days total), wound monitoring.\n"
                "Community physiotherapy: twice weekly until independent mobilisation achieved."
            ),
        },
    },
    {
        "filename": "report_009_sepsis.pdf",
        "patient_id": "SYN-2024-009",
        "date": "2024-07-14",
        "report_type": "ICU Admission Note",
        "sections": {
            "PATIENT INFORMATION": (
                "Patient ID: SYN-2024-009\n"
                "Date of Birth: 1972-08-19  |  Sex: Male  |  Age: 51\n"
                "ICU Admission Date: 2024-07-14  |  Time: 03:45\n"
                "Admitting Intensivist: Dr. R. Messaoudi"
            ),
            "CHIEF COMPLAINT": (
                "Transfer from general ward with deteriorating clinical status: fever 39.8°C, "
                "HR 124 bpm, BP 82/50 mmHg despite 2L IV fluid bolus, RR 28/min, SpO2 91% on room air. "
                "Background: known intra-abdominal abscess post-appendicectomy (Day 5)."
            ),
            "DIAGNOSIS": (
                "Septic shock secondary to intra-abdominal infection (post-operative).\n"
                "SOFA score on ICU admission: 9.\n"
                "Suspected organisms: Gram-negative enteric flora (E. coli, Klebsiella).\n"
                "Source: Residual intra-abdominal collection — IR drainage arranged."
            ),
            "LABORATORY RESULTS": (
                "Critical values on admission:\n"
                "  Lactate: 4.8 mmol/L  (Reference: <2.0)  [CRITICALLY HIGH — tissue hypoperfusion]\n"
                "  WBC: 22.4 × 10⁹/L  [HIGH]  |  Neutrophils: 19.8 × 10⁹/L  [HIGH]\n"
                "  CRP: 312 mg/L  [HIGH]  |  Procalcitonin: 48 ng/mL  [CRITICALLY HIGH]\n\n"
                "Organ Function:\n"
                "  Creatinine: 2.8 mg/dL  (baseline 0.9)  [ACUTE KIDNEY INJURY — Stage 2]\n"
                "  Bilirubin: 3.4 mg/dL  [HIGH]  |  ALT: 88 U/L  [HIGH]\n"
                "  Platelets: 88 × 10⁹/L  [LOW — DIC risk]  |  INR: 1.8  [HIGH]\n\n"
                "Blood cultures: × 2 sets drawn pre-antibiotic. Results pending."
            ),
            "IMAGING": (
                "CT Abdomen/Pelvis (2024-07-14 02:30): Multiloculated collection in right iliac fossa "
                "measuring 6.2 × 4.8 cm with rim enhancement and internal gas — consistent with abscess. "
                "No free perforation. Dilated small bowel loops — ileus pattern."
            ),
            "MEDICATIONS": (
                "Initiated immediately (within 1 hour of sepsis recognition — Sepsis-3 bundle):\n"
                "  - Meropenem 1g IV q8h (broad-spectrum — de-escalate per culture results)\n"
                "  - Metronidazole 500 mg IV q8h (anaerobic coverage)\n"
                "  - Noradrenaline infusion titrated to MAP >65 mmHg\n"
                "  - IV fluid resuscitation: 30 mL/kg crystalloid bolus completed\n"
                "  - Hydrocortisone 200 mg/24h IV (refractory vasopressor requirement)\n"
                "  - VTE prophylaxis: Enoxaparin HELD pending platelet recovery"
            ),
            "TREATMENT PLAN": (
                "1. Source control: IR-guided drainage of abdominal collection — booked 06:00.\n"
                "2. Mechanical ventilation if respiratory status deteriorates (currently on high-flow O2).\n"
                "3. Renal: conservative management, monitor urine output hourly. RRT if oliguric.\n"
                "4. Daily SOFA scoring. Antibiotics review at 48–72h pending cultures.\n"
                "5. Nutrition: enteral feeding to commence when haemodynamically stable."
            ),
            "FOLLOW-UP": (
                "ICU daily review. Family meeting arranged for 10:00 (2024-07-14) — prognosis discussion.\n"
                "Predicted ICU stay: 5–7 days if source controlled and cultures sensitive to antibiotics."
            ),
        },
    },
    {
        "filename": "report_010_thyroid_cancer.pdf",
        "patient_id": "SYN-2024-010",
        "date": "2024-08-20",
        "report_type": "Endocrine Surgery Post-Op Report",
        "sections": {
            "PATIENT INFORMATION": (
                "Patient ID: SYN-2024-010\n"
                "Date of Birth: 1983-05-11  |  Sex: Female  |  Age: 41\n"
                "Surgery Date: 2024-08-19  |  Report Date: 2024-08-20\n"
                "Operating Surgeon: Dr. H. Bouzid"
            ),
            "CHIEF COMPLAINT": (
                "Elective total thyroidectomy for papillary thyroid carcinoma (PTC), "
                "right lobe, 1.8 cm, discovered incidentally on neck ultrasound performed "
                "for unrelated cervical lymphadenopathy workup."
            ),
            "DIAGNOSIS": (
                "Papillary Thyroid Carcinoma, Classical variant, right lobe.\n"
                "pT2 N0 M0 (Stage I, age <55 per AJCC 8th edition).\n"
                "Surgical procedure: Total thyroidectomy + right central neck dissection.\n"
                "Final histopathology: PTC 1.8 cm, margins clear, 0/6 lymph nodes involved."
            ),
            "LABORATORY RESULTS": (
                "Post-operative Day 1:\n"
                "  Calcium (corrected): 7.8 mg/dL  (Reference: 8.5–10.5)  [LOW — transient hypoparathyroidism]\n"
                "  PTH: 8 pg/mL  (Reference: 15–65)  [LOW — expected post total thyroidectomy]\n"
                "  TSH: 0.8 mIU/L  (pre-op, normal — now suppressed post-op for thyroid cancer protocol)\n\n"
                "Pre-operative Thyroid Function (baseline):\n"
                "  TSH: 1.2 mIU/L  |  Free T4: 14.8 pmol/L  |  Free T3: 4.9 pmol/L — All Normal\n"
                "  Thyroglobulin (pre-op): 28 ng/mL  (will be used as tumour marker post-op)"
            ),
            "IMAGING": (
                "Pre-op Neck Ultrasound: Right lobe nodule 1.8 cm, TI-RADS 5 (high suspicion). "
                "No suspicious central or lateral neck nodes.\n\n"
                "Post-op Neck Ultrasound: Pending (planned at 6-week follow-up).\n\n"
                "Radioiodine (I-131) ablation: Planned — multidisciplinary team decision pending staging."
            ),
            "MEDICATIONS": (
                "Post-operative:\n"
                "  - Levothyroxine 125 mcg OD PO (TSH suppression target: 0.1–0.5 mIU/L)\n"
                "  - Calcium carbonate 1g TID PO + Calcitriol 0.25 mcg BID PO (post-thyroidectomy hypocalcaemia)\n"
                "  - Paracetamol 1g QID PO × 5 days\n"
                "  - Monitor: Tingling/perioral numbness (hypocalcaemia symptoms)"
            ),
            "TREATMENT PLAN": (
                "1. Levothyroxine TSH-suppression therapy for 1–2 years then reassess.\n"
                "2. Radioiodine ablation (I-131): MDT recommendation pending — likely indicated for completeness.\n"
                "3. Thyroglobulin monitoring as tumour marker (target: undetectable at 6 months).\n"
                "4. Calcium supplementation until PTH recovery confirmed (typically 6–8 weeks).\n"
                "5. Long-term surveillance: Annual neck ultrasound + TSH/Tg for 5 years minimum."
            ),
            "FOLLOW-UP": (
                "Endocrine surgery: 2024-09-03 (2 weeks) — wound, calcium, PTH, Tg levels.\n"
                "Nuclear medicine MDT: 2024-09-10 — radioiodine decision.\n"
                "Endocrinology: Levothyroxine dose titration, long-term surveillance plan."
            ),
        },
    },
]


def _build_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle", parent=base["Title"], fontSize=14, spaceAfter=4
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle", parent=base["Normal"], fontSize=10,
            textColor="#555555", alignment=TA_CENTER, spaceAfter=12
        ),
        "section_header": ParagraphStyle(
            "SectionHeader", parent=base["Heading2"], fontSize=10,
            spaceBefore=10, spaceAfter=4, textColor="#1a3a5c"
        ),
        "body": ParagraphStyle(
            "Body", parent=base["Normal"], fontSize=9, leading=14, spaceAfter=2
        ),
    }


def generate_report(report: dict, output_dir: Path) -> Path:
    output_path = output_dir / report["filename"]
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    styles = _build_styles()
    story = []

    story.append(Paragraph("SYNTHETIC CLINICAL REPORT — NOT REAL PATIENT DATA", styles["subtitle"]))
    story.append(Paragraph(report["report_type"], styles["title"]))
    story.append(Paragraph(f"Report Date: {report['date']}  |  Patient ID: {report['patient_id']}", styles["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=1, color="#1a3a5c", spaceAfter=8))

    for section_name, content in report["sections"].items():
        story.append(Paragraph(section_name, styles["section_header"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color="#cccccc", spaceAfter=4))
        for line in content.split("\n"):
            story.append(Paragraph(line if line.strip() else "&nbsp;", styles["body"]))
        story.append(Spacer(1, 0.3 * cm))

    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width="100%", thickness=1, color="#1a3a5c"))
    story.append(Paragraph(
        "DISCLAIMER: All data in this document is entirely synthetic and fictional. "
        "Generated for AI/ML research and development purposes only. "
        "No real patient information is contained herein.",
        ParagraphStyle("Disclaimer", parent=_build_styles()["body"], fontSize=7, textColor="#888888")
    ))

    doc.build(story)
    return output_path


def generate_all(output_dir: Path | None = None) -> list[Path]:
    if output_dir is None:
        output_dir = Path(__file__).parents[2] / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    for report in REPORTS:
        path = generate_report(report, output_dir)
        print(f"  Generated: {path.name}")
        generated.append(path)
    return generated


if __name__ == "__main__":
    print("Generating synthetic clinical reports...")
    paths = generate_all()
    print(f"\nDone — {len(paths)} reports written to data/raw/")
