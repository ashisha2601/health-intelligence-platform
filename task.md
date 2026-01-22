# Task Checklist: Health Intelligence Platform

## Phase 1: Data Acquisition & Ingestion
- [x] **Setup Project Structure**: Initialize folders for data, models, and app. <!-- id: 0 -->
- [x] **Ingest Clinical Notes**: Write script to download `AGBonnet/augmented-clinical-notes` (HuggingFace). <!-- id: 1 -->
- [x] **Ingest Synthetic Records**: Write script to download/load Kaggle Synthea dataset. <!-- id: 2 -->
- [x] **Ingest Genomic Data**: Load and normalize `combined_asd_genome_dataset.csv` for genetic risk analysis. <!-- id: 9 -->
- [x] **Database Setup**: Design schema to link Patients -> Conditions -> Observations -> Genotypes. <!-- id: 3 -->

## Phase 2: Analysis Modules
- [x] **Refactor Diabetes Model**: Convert the current notebook model into a reusable Python class/function. <!-- id: 4 -->
- [x] **NLP Summarizer**: Build a simple summarizer for the Clinical Notes. <!-- id: 5 -->
- [x] **Trend Analysis Engine**: Create logic to calculate trends (e.g., "A1C increasing over last 3 visits"). <!-- id: 6 -->

## Phase 3: Dashboard Prototype
- [x] **UI Refactor**: Segregate analysis into tabs (Diabetes, Genomics, NLP) and enforce Light Theme. <!-- id: 9 -->
- [x] **Fix Model Training**: Ensure model trains on startup if pickle is missing to enable feature importance. <!-- id: 10 -->
- [x] **Enhance Dashboard Trends**
    - [x] Prioritize multi-visit patients in selector
    - [x] Added visual cues/tips for trend analysis
    - [x] Fixed "Stable" trend logic (switched to per-visit sequence)
- [x] **Integrate Model Insights**
    - [x] Expose feature importance from `DiabetesReadmissionModel`
    - [x] Add "Global Model Insights" tab with Bar Chart
- [x] **Expand Multi-Modal Analysis**
    - [x] Standardize EDA across all 4 notebooks (Genomic, NLP, Synthea, Diabetes)
    - [x] Integrate Synthea (EHR) population stats into Dashboard (New Tab)
    - [x] Visualize Genomic and NLP metrics in Dashboard
    - [x] Restore full Clinical Notes dataset (30k rows) and fix ingestion limits
    - [x] Add Advanced Visualizations (Radar Chart, Gene Score Scatter, Correlation Heatmap)
    - [x] Add NLP Quality (ROUGE) and Entity (NER) Plots
    - [x] Refine Graphs: Add count labels to bar charts (including NLP)
    - [x] Refine Graphs: Pie Chart (Evidence) & Treemap (Risk Gene Hierarchical Map)
    - [x] Refine Graphs: Genomic Box Plot (Horizontal) & Bar Chart (Avg Risk)
    - [x] Refine Graphs: NLP Compression Violin Plot (Interpretability)
    - [x] Add Data Dictionary Tab with schemas (including Clinical Notes)
    - [x] Add File Uploader for Consumer Data
    - [x] Generate Comprehensive README with Development Log

## Phase 4: Final Assignment Polish
- [x] **Business Interpretable Refinement**
    - [x] Standardize Business-Friendly Visualizations (All Tabs)
    - [x] Apply Pastel Color Scheme
- [x] **Gap Fillers (Requirement Compliance)**
    - [x] Implement Vitals Panel (BP/Cholesterol) in Sidebar/Header
    - [x] Add "Medical Imaging" Tab with Mock Analysis
