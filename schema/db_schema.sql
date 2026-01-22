-- Health Intelligence Platform Schema
-- Optimized for SQLite (portable) but PostgreSQL compatible

-- 1. Patients Table (Demographics)
CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    gender TEXT,
    race TEXT,
    birth_date DATE,
    death_date DATE
);

-- 2. Encounters/Admissions (Visits)
-- Links to Diabetes dataset 'encounter_id'
CREATE TABLE IF NOT EXISTS encounters (
    encounter_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    start_date DATETIME,
    end_date DATETIME,
    encounter_type TEXT, -- e.g., 'Inpatient', 'Ambulatory'
    hospital_id INTEGER,
    readmission_status TEXT, -- From Diabetes dataset (<30, >30, NO)
    FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
);

-- 3. Clinical Notes (Unstructured Data)
-- Linked to encounters
CREATE TABLE IF NOT EXISTS clinical_notes (
    note_id TEXT PRIMARY KEY,
    encounter_id TEXT,
    patient_id TEXT,
    note_type TEXT, -- e.g., 'Discharge Summary', 'Nursing Note'
    note_text TEXT, -- Raw text
    summary TEXT,   -- AI-generated summary
    nlp_entities JSON, -- Extracted entities (Meds, Diseases) stored as JSON
    FOREIGN KEY(encounter_id) REFERENCES encounters(encounter_id)
);

-- 4. Observations (Structured Vitals & Labs)
-- Unifies 'max_glu_serum', 'A1Cresult' from Diabetes DS and future Synthea labs
CREATE TABLE IF NOT EXISTS observations (
    observation_id TEXT PRIMARY KEY,
    encounter_id TEXT,
    patient_id TEXT,
    observation_code TEXT, -- LOINC or internal code (e.g., 'A1C', 'HbA1c')
    observation_display TEXT,
    value_numeric REAL,
    value_text TEXT, -- For categorical results like '>8', 'Normal'
    units TEXT,
    timestamp DATETIME,
    FOREIGN KEY(encounter_id) REFERENCES encounters(encounter_id)
);

-- 5. Conditions/Diagnoses
-- Standardized diagnosis list
CREATE TABLE IF NOT EXISTS conditions (
    condition_id TEXT PRIMARY KEY,
    encounter_id TEXT,
    patient_id TEXT,
    code TEXT, -- ICD-9 or ICD-10 code
    description TEXT,
    rank INTEGER, -- 1=Primary, 2=Secondary, etc.
    FOREIGN KEY(encounter_id) REFERENCES encounters(encounter_id)
);

-- 6. Genomic Reference (Knowledge Base)
-- From 'combined_asd_genome_dataset.csv'
CREATE TABLE IF NOT EXISTS genomic_ref (
    gene_symbol TEXT PRIMARY KEY,
    chromosome TEXT,
    genetic_category TEXT, -- 'Syndromic', 'Rare Single Gene Mutation'
    gene_score REAL,
    is_asd INTEGER, -- Boolean flag
    description TEXT
);

-- 7. Patient Genotypes (Patient-Specific Variations)
-- Maps patients to risks found in genomic_ref
CREATE TABLE IF NOT EXISTS patient_genotypes (
    patient_genotype_id TEXT PRIMARY KEY,
    patient_id TEXT,
    gene_symbol TEXT,
    variant_type TEXT, -- 'SNP', 'CNV'
    zygosity TEXT, -- 'Heterozygous', 'Homozygous'
    FOREIGN KEY(patient_id) REFERENCES patients(patient_id),
    FOREIGN KEY(gene_symbol) REFERENCES genomic_ref(gene_symbol)
);
