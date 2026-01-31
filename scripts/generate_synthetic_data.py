import pandas as pd
import numpy as np
import os

def generate_synthetic_diabetes(path, n=500):
    """Generates synthetic diabetes readmission data with longitudinal histories."""
    np.random.seed(42)
    
    # Logic: 1/5th of n will be unique patients to ensure history
    n_unique = n // 5
    unique_pats = np.random.randint(100000, 999999, n_unique)
    
    # Distribute n rows among unique patients
    rows = []
    base_enc_id = 500000000
    
    for i, pat_id in enumerate(unique_pats):
        # Each patient gets 1 to 15 visits
        n_visits = np.random.randint(1, 15)
        
        # Base stats for this patient to keep them consistent
        race = np.random.choice(['Caucasian', 'AfricanAmerican', 'Asian', 'Hispanic', 'Other'])
        gender = np.random.choice(['Female', 'Male'])
        age = np.random.choice(['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'])
        
        # Initial health state
        base_lab_intensity = np.random.randint(20, 80)
        improving = np.random.choice([True, False]) # Some get better, some worse
        
        for v in range(n_visits):
            # Evolve labs over visits (Trend Simulation)
            drift = -3 if improving else 3
            lab_val = max(1, min(100, base_lab_intensity + (v * drift) + np.random.randint(-5, 5)))
            
            row = {
                'encounter_id': base_enc_id + len(rows),
                'patient_nbr': pat_id,
                'race': race,
                'gender': gender,
                'age': age,
                'time_in_hospital': np.random.randint(1, 14),
                'num_lab_procedures': lab_val,
                'num_procedures': np.random.randint(0, 6),
                'num_medications': np.random.randint(1, 60),
                'number_outpatient': np.random.randint(0, 10),
                'number_emergency': np.random.randint(0, 5),
                'number_inpatient': np.random.randint(0, 12),
                'diag_1': np.random.choice(['250', '428', '414', '715', '401']),
                'diag_2': np.random.choice(['250', '428', '414', '715', '401']),
                'diag_3': np.random.choice(['250', '428', '414', '715', '401']),
                'number_diagnoses': np.random.randint(1, 16),
                'max_glu_serum': np.random.choice(['None', '>200', '>300', 'Norm']),
                'A1Cresult': np.random.choice(['None', '>7', '>8', 'Norm']),
                'insulin': np.random.choice(['No', 'Up', 'Steady', 'Down']),
                'change': np.random.choice(['Ch', 'No']),
                'readmitted': np.random.choice(['<30', '>30', 'NO']),
                'diabetesMed': np.random.choice(['Yes', 'No'])
            }
            # Add placeholders
            for col in ['weight', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'payer_code', 'medical_specialty',
                        'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 
                        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 
                        'examide', 'citoglipton', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 
                        'metformin-rosiglitazone', 'metformin-pioglitazone']:
                row[col] = '?'
            
            rows.append(row)
            if len(rows) >= n: break
        if len(rows) >= n: break

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Generated {len(df)} rows of longitudinal clinical data at {path}")

def generate_synthetic_genomic(path, n=50):
    """Generates synthetic genomic ASD risk data."""
    np.random.seed(42)
    genes = ['SHANK3', 'ARID1B', 'SYNGAP1', 'SCN2A', 'DYRK1A', 'ADNP', 'POGZ', 'CHD8', 'ASH1L', 'GRIN2B']
    data = {
        'gene_symbol': np.random.choice(genes, n),
        'chromosome': np.random.choice([str(i) for i in range(1, 23)] + ['X', 'Y'], n),
        'gene-score': np.random.choice([0.0, 1.0, 2.0, 3.0], n),
        'is_asd': np.random.choice([0, 1], n),
        'genetic-category': np.random.choice(['Category 1', 'Category 2', 'Category 3', 'Non-ASD Gene'], n),
        'number-of-reports': np.random.randint(0, 50, n),
        'gene_length': np.random.randint(1000, 500000, n),
        'description': 'Synthetic ASD related gene description'
    }
    # Add other headers
    for col in ['gene_entries_count', 'avg_exon_count', 'map_location', 'has_omim', 'is_plus_strand', 'genomic_start', 'genomic_end', 'syndromic', 'chromosome_encoded']:
        data[col] = 0
        
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    print(f"Generated {n} rows of genomic data at {path}")

def generate_synthetic_notes(path, n=10):
    """Generates synthetic clinical notes."""
    np.random.seed(42)
    notes = [
        "Patient presented with polyuria and polydipsia. Blood glucose elevated at 250 mg/dL. Adjusted insulin dose.",
        "Follow up for diabetic ketoacidosis. Patient reports better adherence to diet but struggles with exercise.",
        "Routine checkup. A1C remains high at 8.2%. Discussed adding metformin to the regimen.",
        "Emergency admission for hypoglycemia. Patient reported skipping meals while on steady insulin dose."
    ]
    data = {
        'idx': np.arange(n),
        'note': np.random.choice(notes, n),
        'summary': 'Synthetic AI generated summary here.',
        'full_note': 'Synthetic Full Clinical Narrative...',
        'conversation': 'Synthetic Conversation Log...'
    }
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    print(f"Generated {n} clinical notes at {path}")

if __name__ == "__main__":
    base_path = "/Users/ashishasharma/Desktop/smarttech/healthcare-multimodal"
    out_dir = os.path.join(base_path, "data/synthetic_demo")
    os.makedirs(out_dir, exist_ok=True)
    
    generate_synthetic_diabetes(os.path.join(out_dir, "diabetes_data.csv"))
    generate_synthetic_genomic(os.path.join(out_dir, "combined_asd_genome_dataset.csv"))
    generate_synthetic_notes(os.path.join(out_dir, "clinical_notes_small.csv"))
    print(f"\nSuccess: Synthetic datasets created in {out_dir}")
