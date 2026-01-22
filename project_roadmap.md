# Health Intelligence Platform - Project Roadmap

## 1. Executive Summary
**Goal**: Build a consumer-facing application that aggregates multi-modal health data (EHR, Imaging, Genomics, Clinical Notes), performs AI-driven inference, and visualizes trends/predictions on a user-friendly dashboard.

## 2. System Architecture

### Phase 1: Data Ingestion & Database Construction (The Foundation)
**Objective**: Create a unified patient health record (PHR).
*   **Data Sources**:
    1.  **Structured Data**: Lab results, vitals, demographics (Source: Diabetes Dataset, Synthea).
    2.  **Unstructured Text**: Clinical notes, discharge summaries (Source: PMC-Patients, NoteChat).
    3.  **Medical Imaging**: X-Rays, CT Scans, MRIs.
    4.  **Genomics**: DNA/Gene data (Source: ASD Genome dataset).
*   **Key Tasks**:
    *   [ ] **ETL Pipeline**: Build scripts to ingest CSV, JSON, PDF (OCR), and DICOM files.
    *   [ ] **Database Schema**: Design a SQL (PostgreSQL) schema for structured data and NoSQL (MongoDB/VectorDB) for unstructured documents/embeddings.
    *   [ ] **Dataset Acquisition**: Download and inspect `AGBonnet/augmented-clinical-notes` and `imtkaggleteam/synthetic-medical-dataset`.

### Phase 2: Diagnostic Analysis (The Intelligence Layer)
**Objective**: Derive insights from raw data.
*   **Module A: Structured Risk Prediction**
    *   Use the **Diabetes Readmission Model** (already working on) as the prototype.
    *   Expand to Hypertension, Cholesterol trends using Synthea data.
*   **Module B: Clinical NLP (Text)**
    *   Tasks: Summarization of doctor notes, Entity Extraction (Medications, Diseases).
    *   Models: Fine-tune BioBERT or use LLMs (Llama 3 Med / GPT-4) on the *Augmented Clinical Notes* dataset.
*   **Module C: Medical Imaging (Vision)**
    *   Tasks: Detect anomalies in X-Rays/CTs.
    *   Models: ResNet/ViT pre-trained on medical imaging datasets (e.g., CheXpert).
*   **Module D: Genomics (Bioinformatics)**
    *   **Goal**: Flag genetic risk factors for ASD and other conditions.
    *   **Input**: `combined_asd_genome_dataset.csv` (Columns: `gene_symbol`, `chromosome`, `genetic-category`, `gene-score`, `is_asd`).
    *   **Action**: Map patient genetic markers to this reference tables to calculate a polygenic risk score.

### Phase 3: Consumer Dashboard (The User Interface)
**Objective**: "Precision presentation of trends."
*   **Features**:
    *   **Timeline View**: Longitudinal view of A1C, BP, BMI.
    *   **Risk Scorecards**: "High risk for Diabetes", "Moderate risk for CVD".
    *   **Plain English Summaries**: Translating medical jargon from notes into patient-friendly language.
*   **Tech Stack**: Streamlit (fastest prototype) or React + FastAPI (production).

---

## 3. Dataset Integration Plan

### A. Biomedical / Text Data
**Source**: [AGBonnet/augmented-clinical-notes](https://huggingface.co/datasets/AGBonnet/augmented-clinical-notes)
*   **Use Case**: Training a "Doctor AI Assistant" to summarize history and explain diagnoses.
*   **Action**: Write script to download Parquet files and load into Pandas/DB.

### B. Synthetic Medical History
**Source**: [Kaggle - Synthetic Medical Dataset (Synthea)](https://kaggler.com/datasets/imtkaggleteam/synthetic-medical-dataset)
*   **Use Case**: Generates realistic longitudinal data (A1C over 10 years, BP trends) which the diabetes dataset lacks.
*   **Action**: Ingest into SQL to simulate patient history.

### C. Genomic Data
**Source**: `combined_asd_genome_dataset.csv` (Your local file)
*   **Use Case**: Specialized genetic risk factors.

---

## 4. Addressing Your Questions
*   **"Is the diabetes data enough?"**: No. It is excellent for *one specific prediction* (readmission), but for a "consumer health app", you need the **Synthea** data to show long-term trends (A1C history, not just one point) and broader conditions.
*   **"Is the notebook analysis correct?"**: Yes, it followed a standard Data Science lifecycle (cleaning -> EDA -> model). However, for an *App*, we need to convert that analysis into a **Pipeline**: `Raw Data -> Cleaning Function -> Model.predict() -> JSON Output`. We move from "exploring" to "engineering".

