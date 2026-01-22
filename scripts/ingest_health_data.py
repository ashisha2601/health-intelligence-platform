import pandas as pd
from datasets import load_dataset
import os

def ingest_clinical_notes():
    print("--- Ingesting Clinical Notes (Hugging Face) ---")
    try:
        # Load the dataset from Hugging Face
        # Using 'AGBonnet/augmented-clinical-notes' as discussed
        # Load the dataset from Hugging Face
        # Using 'AGBonnet/augmented-clinical-notes' as discussed
        dataset = load_dataset("AGBonnet/augmented-clinical-notes", split="train")
        
        print("Dataset loaded. Converting to DataFrame...")
        df = dataset.to_pandas()
        
        print(f"Successfully retrieved {len(df)} rows.")
        
        # SAVE the file
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        output_path = os.path.join(output_dir, "clinical_notes.csv")
        df.to_csv(output_path, index=False)
        print(f"✅ Saved Clinical Notes to: {output_path}")

        print("Columns:", df.columns.tolist())
        return df
    except Exception as e:
        print(f"Error ingesting clinical notes: {e}")
        return None

def ingest_genomic_data(filepath):
    print(f"\n--- Ingesting Genomic Data ({filepath}) ---")
    try:
        if not os.path.exists(filepath):
            print(f"File not found at: {filepath}")
            return None
            
        df = pd.read_csv(filepath)
        print(f"Successfully loaded {len(df)} rows.")
        print("Columns:", df.columns.tolist())
        
        # Specialized preview for the columns we identified in planning
        target_cols = ['gene_symbol', 'chromosome', 'genetic-category', 'gene-score', 'is_asd']
        available_cols = [c for c in target_cols if c in df.columns]
        
        print("\nPreview (Target Columns):")
        print(df[available_cols].head())
        return df
    except Exception as e:
        print(f"Error ingesting genomic data: {e}")
        return None



import kagglehub

def ingest_synthea():
    print("\n--- Ingesting Synthetic Records (Synthea) ---")
    try:
        # Download latest version
        path = kagglehub.dataset_download("imtkaggleteam/synthetic-medical-dataset")
        print("Path to dataset files:", path)
        
        # List files in the downloaded directory to find the CSV
        files = os.listdir(path)
        print("Files found:", files)
        
        # Try to find a relevant CSV (e.g., patient records)
        # Synthea usually has 'patients.csv', 'observations.csv', etc.
        # We'll look for any csv to preview
        csv_files = [f for f in files if f.endswith('.csv')]
        
        if csv_files:
            target_csv = os.path.join(path, csv_files[0])
            print(f"Previewing first CSV found: {target_csv}")
            df = pd.read_csv(target_csv)
            
            # SAVE the file so the dashboard can find it
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
            output_path = os.path.join(output_dir, "synthea_ehr_data.csv")
            df.to_csv(output_path, index=False)
            print(f"✅ Saved Synthea data to: {output_path}")
            
            print("Columns:", df.columns.tolist())
            print(df.head(2))
            return df
        else:
            print("No CSV files found in the dataset folder.")
            return None

    except Exception as e:
        print(f"Error ingesting Synthea data: {e}")
        print("NOTE: You may need to authenticate with Kaggle. Run 'kagglehub.login()' in a separate script if this fails due to auth.")
        return None

def main():
    # 1. Clinical Notes
    notes_df = ingest_clinical_notes()
    
    # 2. Genomic Data (Local File)
    # Relative path from scripts/ to data/
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    genomic_path = os.path.join(base_path, "data", "combined_asd_genome_dataset.csv") 
    genomic_df = ingest_genomic_data(genomic_path)

    # 3. Synthea (Kaggle)
    synthea_df = ingest_synthea()

if __name__ == "__main__":
    main()
