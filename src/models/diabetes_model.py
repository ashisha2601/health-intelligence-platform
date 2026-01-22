import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import joblib
import os

class DiabetesReadmissionModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs', random_state=42)
        self.scaler = StandardScaler()
        self.target_encoder_map = {}
        
        self.numerical_features = []
        self.categorical_features = []
        self.high_cardinality_features = ['payer_code', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3']
        self.low_card_categoricals = []
        self.feature_names = [] 

    def preprocess(self, df):
        df_clean = df.copy()
        
        rename_map = {
            'time_in_hospital': 'length_of_stay_days',
            'num_lab_procedures': 'lab_procedures_count',
            'num_procedures': 'procedures_count',
            'num_medications': 'medication_count',
            'number_outpatient': 'outpatient_visits',
            'number_emergency': 'emergency_visits',
            'number_inpatient': 'inpatient_visits'
        }
        df_clean = df_clean.rename(columns=rename_map)
        df_clean = df_clean.replace(['?', 'Unknown', 'Unknown/Invalid'], np.nan).infer_objects(copy=False)
        
        leakage_cols = ['discharge_disposition_code', 'discharge_disposition_id']
        df_clean = df_clean.drop(columns=[c for c in leakage_cols if c in df_clean.columns])
        
        return df_clean

    def feature_engineering(self, df):
        df_eng = df.copy()
        
        if 'number_diagnoses' in df_eng.columns:
             df_eng['total_diagnoses'] = df_eng['number_diagnoses']
        elif 'num_diagnoses' in df_eng.columns:
             df_eng['total_diagnoses'] = df_eng['num_diagnoses']
             
        if 'medication_count' in df_eng.columns:
            df_eng['high_medication_count'] = (df_eng['medication_count'] > 20).astype(int)
            
        visit_cols = ['emergency_visits', 'inpatient_visits', 'outpatient_visits']
        if all(col in df_eng.columns for col in visit_cols):
            df_eng['total_prior_visits'] = df_eng[visit_cols].sum(axis=1)
            df_eng['frequent_visitor'] = (df_eng['total_prior_visits'] > 5).astype(int)
            df_eng['has_prior_inpatient'] = (df_eng['inpatient_visits'] > 0).astype(int)
            
        if 'insulin' in df_eng.columns:
            df_eng['on_insulin'] = (df_eng['insulin'] != 'No').astype(int)
        if 'metformin' in df_eng.columns:
            df_eng['on_metformin'] = (df_eng['metformin'] != 'No').astype(int)
        if 'change' in df_eng.columns:
            df_eng['medication_changed'] = (df_eng['change'] == 'Ch').astype(int)
            
        if 'A1Cresult' in df_eng.columns:
            df_eng['A1C_tested'] = (df_eng['A1Cresult'] != 'None').astype(int)
            df_eng['A1C_abnormal'] = (df_eng['A1Cresult'].isin(['>7', '>8'])).astype(int)
            
        if 'max_glu_serum' in df_eng:
            df_eng['glucose_tested'] = (df_eng['max_glu_serum'] != 'None').astype(int)
            df_eng['glucose_abnormal'] = (df_eng['max_glu_serum'].isin(['>200', '>300'])).astype(int)
            
        if 'age' in df_eng.columns:
            age_mapping = {
                '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
                '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
                '[80-90)': 85, '[90-100)': 95
            }
            df_eng['age_numeric'] = df_eng['age'].map(age_mapping).fillna(55) # Median fallback
            df_eng['elderly'] = (df_eng['age_numeric'] >= 65).astype(int)
            
        complexity_cols = ['total_diagnoses', 'medication_count', 'total_prior_visits', 'length_of_stay_days']
        existing = [c for c in complexity_cols if c in df_eng.columns]
        if existing:
            comps = [df_eng[c]/df_eng[c].max() if df_eng[c].max() > 0 else df_eng[c] for c in existing]
            df_eng['care_complexity_score'] = pd.concat(comps, axis=1).mean(axis=1)
            
        return df_eng

    def _target_encode_fit(self, series, target, smoothing=10.0):
        global_mean = target.mean()
        stats = pd.DataFrame({'feature': series, 'target': target}).groupby('feature')['target'].agg(['count', 'mean'])
        smoothed = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
        return smoothed.to_dict(), global_mean

    def _target_encode_transform(self, series, mapping, global_mean):
        return series.map(mapping).fillna(global_mean)

    def prepare_data(self, df, target_col='readmitted'):
        df_p = self.preprocess(df)
        df_p = self.feature_engineering(df_p)
        
        if target_col not in df_p.columns:
            raise ValueError(f"Target {target_col} missing")
            
        y = (df_p[target_col] == '<30').astype(int)
        X = df_p.drop(columns=[target_col, 'encounter_id', 'patient_nbr', 'readmit_30_days'], errors='ignore')
        
        self.numerical_features = X.select_dtypes(include=['number']).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.low_card_categoricals = [c for c in self.categorical_features if c not in self.high_cardinality_features]
        high_card_exists = [c for c in self.high_cardinality_features if c in self.categorical_features]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        # Encoding
        X_train_enc = X_train.copy()
        X_test_enc = X_test.copy()
        
        for col in high_card_exists:
            mapping, mean = self._target_encode_fit(X_train[col], y_train)
            self.target_encoder_map[col] = {'map': mapping, 'mean': mean}
            X_train_enc[col+'_enc'] = self._target_encode_transform(X_train[col], mapping, mean)
            X_test_enc[col+'_enc'] = self._target_encode_transform(X_test[col], mapping, mean)
            X_train_enc.drop(columns=[col], inplace=True)
            X_test_enc.drop(columns=[col], inplace=True)

        X_train_dummies = pd.get_dummies(X_train_enc, columns=self.low_card_categoricals, drop_first=True)
        X_test_dummies = pd.get_dummies(X_test_enc, columns=self.low_card_categoricals, drop_first=True)
        
        # Align columns
        for c in set(X_train_dummies.columns) - set(X_test_dummies.columns):
            X_test_dummies[c] = 0
        X_test_dummies = X_test_dummies[X_train_dummies.columns]
        
        self.feature_names = X_train_dummies.columns.tolist()
        
        # Impute
        for col in self.feature_names:
            if X_train_dummies[col].isna().any():
                val = X_train_dummies[col].median()
                X_train_dummies[col].fillna(val, inplace=True)
                X_test_dummies[col].fillna(val, inplace=True)

        X_train_scaled = self.scaler.fit_transform(X_train_dummies)
        X_test_scaled = self.scaler.transform(X_test_dummies)
        
        return X_train_scaled, X_test_scaled, y_train, y_test

    def train(self, X_train, y_train):
        print("Training model...")
        self.model.fit(X_train, y_train)
        
    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))
        
    def save(self, filepath):
        joblib.dump(self, filepath)
        
    def get_feature_importance(self):
        if not hasattr(self.model, 'coef_'): return None
        coeffs = self.model.coef_[0]
        return pd.DataFrame({'feature': self.feature_names, 'importance': abs(coeffs)}).sort_values('importance', ascending=False)

if __name__ == "__main__":
    pass
