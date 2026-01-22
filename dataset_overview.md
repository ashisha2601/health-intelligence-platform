# Dataset Overview: Diabetes Readmission Prediction

## 1. Dataset Summary
- **Dataset Name**: UCI Diabetes 130-US Hospitals (1999-2008)
- **Objective**: Predict 30-day hospital readmission for diabetic patients.
- **Total Records (Rows)**: 101,766
- **Total Features (Columns)**: 50
- **Memory Usage**: ~192.87 MB

## 2. Feature Structure & Categorization
The dataset contains a mix of demographic, clinical, and administrative features.

### **Demographics**
- `race`: Patient's race (Caucasian, AfricanAmerican, etc.)
- `gender`: Male/Female
- `age`: Age range (e.g., [70-80))
- `weight`: Patient weight (Highly missing)

### **Clinical/Diagnostic**
- `diag_1`, `diag_2`, `diag_3`: ICD-9 diagnosis codes (Primary, Secondary, Tertiary)
- `number_diagnoses`: Total number of diagnoses entered
- `max_glu_serum`: Glucose serum test result
- `A1Cresult`: HbA1c test result

### **Treatment/Medication**
- `num_medications`: Number of distinct medications administered
- `diabetesMed`: Indicates if any diabetes medication was prescribed
- `change`: Indicates if there was a change in diabetes medication
- **Specific Medications**: 24 distinct features for medications like `metformin`, `insulin`, `glipizide`, etc.

### **Utilization (Hospital History)**
- `time_in_hospital`: Length of stay (days)
- `num_lab_procedures`: Number of lab tests performed
- `num_procedures`: Number of procedures (non-lab) performed
- `number_outpatient`: Number of outpatient visits in preceding year
- `number_emergency`: Number of emergency visits in preceding year
- `number_inpatient`: Number of inpatient visits in preceding year
- `admission_type_id`, `admission_source_id`, `discharge_disposition_id`: Administrative identifiers

### **Target Variable**
- `readmitted`: The readmission status of the patient.
    - **NO**: No readmission (~54%)
    - **>30**: Readmission after 30 days (~35%)
    - **<30**: Readmission within 30 days (~11%)

## 3. Data Quality & Missing Values
Significant missingness was identified in several columns, often encoded as `?`.

| Column              | Missing Count | Percentage | Note                                  |
|---------------------|---------------|------------|---------------------------------------|
| **weight**          | 98,569        | ~97%       | Too scarce to be useful for most models |
| **max_glu_serum**   | 96,420        | ~95%       | Likely indicated test not ordered     |
| **A1Cresult**       | 84,748        | ~83%       | Test not performed for most patients  |
| **medical_specialty**| 49,949       | ~49%       | Specialty of admitting physician often missing |
| **payer_code**      | 40,256        | ~40%       | Insurance info missing                |
| **race**            | 2,273         | ~2%        | Minor missingness                     |
| **diag_3**          | 1,423         | ~1%        | Tertiary diagnosis missing for some   |

**Handling Strategy identified in notebook:**
- High missingness in `weight` makes it difficult to use directly.
- Missing values in test results (`A1Cresult`) are often informative (e.g., "Not Tested" is a valid category).

## 4. Target Variable Distribution
The dataset is imbalanced but not extremely so.
- **Majority Class (NO):** 54,864 (53.9%)
- **Minority Class (<30 days - Critical):** 11,357 (11.2%)
- **Intermediate Class (>30 days):** 35,545 (34.9%)

*Note: For binary classification focuses on <30 days readmission, this presents a significant class imbalance (approx 1:9).*

## 5. Potential Data Leakage
The analysis flagged potential leakage risks:
- **`discharge_disposition_id`**: Can contain codes that relate to death or hospice, which would preclude readmission.
- **Post-discharge data**: Any data not available at the moment of discharge decision should be excluded.
