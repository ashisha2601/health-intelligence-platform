# Health Intelligence Platform

A comprehensive, multi-modal healthcare analysis dashboard integrating Clinical EHR, Genomic, and NLP data to provide predictive insights and personalized health intelligence.

## 🚀 Key Features

*   **Consumer-Facing Dashboard**: A user-friendly Streamlit interface for exploring patient health data.
*   **Predictive Diagnostics**: Machine Learning model (`DiabetesReadmissionModel`) to predict 30-day hospital readmission risk.
*   **Genomic Risk Analysis**: Heuristic and evidence-based scoring of ASD-related risk genes (SFARI Gene Score).
*   **Clinical NLP**: Automated summarization of unstructured clinical notes using Transformer models (`DistilBART`).
*   **Trend Analysis**: Longitudinal tracking of key vital signs (e.g., A1C, Glucose) with automated trend detection (Increasing/Decreasing/Stable).
*   **Data Upload**: End-users can upload their own CSV health records directly via the sidebar.

## 📁 Project Structure

```bash
healthcare-multimodal/
├── data/                       # Raw datasets (Clinical, Genomic, Synthetic)
├── src/
│   ├── app/
│   │   └── dashboard.py        # Main Streamlit Application
│   ├── models/
│   │   ├── diabetes_model.py   # Readmission Prediction Model (SKLearn)
│   │   ├── nlp_summarizer.py   # Clinical Note Summarizer (HuggingFace)
│   │   └── trend_analyzer.py   # Longitudinal Data Analysis Engine
├── schema/
│   └── db_schema.sql           # Database Design (Patients -> Encounters -> Genotypes)
└── scripts/
    └── ingest_health_data.py   # Data Ingestion & Processing Pipeline
```

## 🛠️ Step-by-Step Development Log

### Phase 1: Data Infrastructure
1.  **Ingestion Pipelines**: Created `ingest_health_data.py` to:
    *   Download Clinical Notes from HuggingFace (`AGBonnet/augmented-clinical-notes`).
    *   Download Synthetic EHR data from Kaggle (`synthetic-medical-dataset`).
    *   Load Genomic Risk data (`combined_asd_genome_dataset.csv`).
2.  **Database Design**: Designed a robust SQL schema `db_schema.sql` linking Patients, Encounters, Observations, and Genotypes.

### Phase 2: AI & Analysis Modules
3.  **Readmission Model**: Built `DiabetesReadmissionModel` using Logistic Regression.
    *   **Features**: `time_in_hospital`, `num_medications`, `number_diagnoses`, etc.
    *   **Output**: Probability of readmission within 30 days.
4.  **Trend Engine**: Developed `TrendAnalyzer` to detect patterns in time-series lab data.
    *   **Logic**: Calculates slope and consistency to flag "Increasing Risk".
5.  **NLP Summarizer**: Integrated `ClinicalSummarizer` using `sshleifer/distilbart-cnn-12-6` to condense lengthy discharge summaries into key points.

### Phase 3: Dashboard Prototype
6.  **Interactive UI**: Built `src/app/dashboard.py` using Streamlit.
    *   **Tabs**: Segregated analysis into "Diabetes Analysis", "Genomic Risk", "Clinical Notes", and "Model Insights".
    *   **Patient Selector**: Sidebar widget to switch between patient profiles dynamically.
7.  **Visualizations**: Integrated `Plotly` for interactive charting.
    *   **Risk Radar**: Radar chart comparing patient metrics to population averages.
    *   **Genomic Treemap**: Hierarchical map of Genetic Risk Categories by Reliability Score.
    *   **NLP Metrics**: Bar charts showing token reduction and compression efficiency.

### Phase 4: Refinement & Consumer Features
8.  **Model Insights**: Added deep-dive visualizations for explainability.
    *   **Feature Importance**: Top 10 predictors driving the readmission model.
    *   **Data Dictionary**: Interactive schema documentation for all datasets.
9.  **File Uploader**: Added a sidebar widget allowing users to assume control by uploading their own CSV datasets (Clinical or Genomic), effectively making the platform a "Bring Your Own Data" solution.
10. **Data Labeling**: Enhanced all charts with explicit count/percentage labels for better interpretability.

## 🌐 Live Deployment
The platform is live and can be accessed at:
[health-intelligence-platform-zjajfyvepkfwwey7keapppx.streamlit.app](https://health-intelligence-platform-zjajfyvepkfwwey7keapppx.streamlit.app/)

## 🏃‍♂️ How to Run

1.  **Install Dependencies**:
    ```bash
    pip install pandas numpy streamlit plotly scikit-learn transformers torch
    ```
2.  **Launch Dashboard**:
    ```bash
    streamlit run src/app/dashboard.py
    ```
3.  **Access**: Open `http://localhost:8501` in your browser.

## 🔮 Future Roadmap
*   **Live Inference Endpoint**: Deploy models as REST APIs (FastAPI) for real-time integration.
*   **FHIR Integration**: Standardize data ingestion using HL7 FHIR resources.
*   **Cloud Deployment**: [DONE] Successfully deployed to Streamlit Cloud.
