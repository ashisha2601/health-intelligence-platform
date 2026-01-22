# Setup Instructions

## Prerequisites
- Python 3.8 or higher
- pip package manager

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## Dataset

Download the UCI Diabetes 130-US Hospitals dataset:
- URL: https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008
- Place the `diabetic_data.csv` file in `data/raw/` directory

## Running the Analysis

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `healthcare_readmission_analysis.ipynb`

3. Run all cells to execute the full analysis

## Notes

- The notebook includes sample data generation for demonstration
- Replace the sample data loading section with actual dataset loading
- Adjust file paths as needed for your environment
