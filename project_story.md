# 🏥 Health Intelligence Platform: The Unified "Project Story"

## 📖 The Mission
Modern healthcare data is siloed: Electronic Health Records (EHR) live in one system, Genomic data in another, and unstructured Clinical Notes in a third. This fragmentation makes it impossible for patients to see the "Full Picture" of their health.

**Our Goal**: Develop a single, consumer-facing **"Health Intelligence Platform"** that ingests multi-modal data (Text + Numbers + Images + Genes), runs advanced AI inference, and presents it in a beautiful, business-friendly Dashboard.

---

## 🏗️ Technical Architecture (The "How")

### 1. Data Ingestion Layer (The Foundation)
We started by building a robust pipeline (`ingest_health_data.py`) to collect disparate real & synthetic datasets:
*   **Structured Data**: 100k+ rows of Diabetes Encounters (Kaggle) & Synthea EHR records.
*   **Unstructured Text**: Clinical Notes from HuggingFace (`augmented-clinical-notes`).
*   **Genomic Data**: ASD-linked gene variants (`combined_asd_genome_dataset.csv`).
*   **Imaging**: Support for X-Ray/CT scans (DICOM/PNG).

### 2. Multi-Modal AI Engine (The Brain)
We didn't just display data; we applied specific AI techniques for each modality:
*   **Predictive Analytics (Structured)**: Trained a **Logistic Regression** model (`DiabetesReadmissionModel`) to predict 30-day hospital readmission risk based on `time_in_hospital`, `num_medications`, and `diagnosis_count`.
*   **Natural Language Processing (Text)**: Deployed a **DistilBART Transformer** (`sshleifer/distilbart-cnn-12-6`) to automatically summarize verbose doctor's notes into concise, readable abstracts.
*   **Computer Vision (Images)**: Integrated a mock **Anomaly Detection** pipeline to flagging potential issues in uploaded X-Rays.
*   **Statistical Trend Engine**: Built a custom `TrendAnalyzer` using **Linear Regression Slopes** to detect if vitals (like Lab Procedures) are strictly "Increasing" or "Decreasing" over time.

### 3. The Dashboard (The Experience)
We chose **Streamlit** for the frontend to bridge the gap between Data Science and Consumer UX.
*   **Design Philosophy**: "Business-Friendly & Interpretable".
    *   **Pastel Aesthetic**: We moved away from "Traffic Light Scares" (Red/Green) to a calming Pastel Palette (`#aec7e8`, `#ffbb78`) to reduce patient anxiety.
    *   **Plain English**: We renamed technical metrics (e.g., "ROUGE Score") to actionable questions (e.g., *"Does it save reading time?"*).
*   **Key Features**:
    *   **Comparison Engine**: Visualizes "Patient vs Population" to benchmark costs and risks.
    *   **Bring Your Own Data**: A universal **File Uploader** allows any user to plug their CSVs/Images into our AI engine instantly.

---

## 🔬 Scientific Concepts Used

### 🩺 Statistical & Machine Learning
*   **Logistic Regression**: For binary classification (Readmitted vs Not).
*   **Transformers (Enc-Dec)**: For abstractive text summarization.
*   **Feature Importance**: calculating `coef_` to explain *why* a patient is high risk (e.g., *"High Number of Inpatient Visits"*).
*   **Linear Trends**: Calculating the slope ($m$) of $y = mx + b$ to determine health trajectory.

### 🧬 Medical Concepts
*   **A1C & Diabetes**: Tracking Glycated Hemoglobin levels to determine diabetic control.
*   **Polypharmacy**: Identifying patients taking >20 medications as a high-risk cohort.
*   **Genomic Linkage**: Mapping Gene Symbols (e.g., `CHD8` - Chromosome 5) to Risk Scores (SFARI) to assess hereditary conditions.
*   **Vitals Monitoring**: Synthesizing Blood Pressure (Systolic/Diastolic) and Cholesterol trends based on clinical risk profiles.

---

## 🌟 The "Journey" (User Flow)
1.  **Ingest**: The app loads the "Hospital Baseline" (EHR Tab).
2.  **Select**: The user picks a Patient ID. The app instantly retrieves their History, Genomics, and Notes.
3.  **Analyze**:
    *   **Diabetes Tab**: Shows the readmission risk gauge.
    *   **Genomic Tab**: Maps their variants to "Hotspots" on chromosomes.
    *   **NLP Tab**: Reads their doctor's note and writes a summary for them.
    *   **Imaging Tab**: They upload an X-Ray, and the AI says "No Anomalies".
4.  **Action**: The user sees the "Vitals Panel" (BP/Cholesterol) and decides to consult a doctor if the trend is "Increasing".

This project transforms raw numbers into **Health Intelligence**.
