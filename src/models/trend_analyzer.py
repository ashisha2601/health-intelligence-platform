
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrendAnalyzer:
    def __init__(self):
        pass

    def analyze_trend(self, dates, values):
        """
        Analyze the trend of a series of values over time.
        
        Args:
            dates (list or pd.Series): Date objects or datetime strings.
            values (list or pd.Series): Numerical values.
            
        Returns:
            dict: {
                'slope': float,
                'direction': str ('Increasing', 'Decreasing', 'Stable'),
                'volatility': float (std dev),
                'r_squared': float,
                'start_value': float,
                'end_value': float
            }
        """
        if len(values) < 2:
            return {
                'slope': 0, 'direction': 'Insufficient Data', 'volatility': 0, 
                'r_squared': 0, 'start_value': values[0] if len(values)>0 else None, 'end_value': values[-1] if len(values)>0 else None
            }
            
        # Robust handling: Treat 'dates' as a sequence scaler (whether it's Sequence ID or Date)
        # We use the RANK/INDEX of the sorted sequence for the slope calculation (Slope per Visit)
        # This avoids issues where int IDs are interpreted as "1970-01-01" timestamps.
        df = pd.DataFrame({'seq': list(dates), 'value': list(values)})
        
        # Sort by the sequence key (Date or Encounter ID)
        df = df.sort_values('seq').dropna()
        
        if len(df) < 2:
             return {'slope': 0, 'direction': 'Insufficient Data', 'volatility': 0, 'r_squared': 0, 'start_value': None, 'end_value': None}

        # Use purely sequential index for X (0, 1, 2...) -> Trend per Visit
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['value'].values
        
        # Fit Linear Regression
        model = LinearRegression()
        model.fit(X, y)
        
        slope = model.coef_[0]
        r_squared = model.score(X, y)
        std_dev = np.std(y)
        
        # Determine direction (thresholds can be tuned)
        # Using a small epsilon for stability
        if slope > 0.001:
            direction = "Increasing"
        elif slope < -0.001:
            direction = "Decreasing"
        else:
            direction = "Stable"
            
        return {
            'slope': slope,
            'direction': direction,
            'volatility': std_dev,
            'r_squared': r_squared,
            'start_value': df['value'].iloc[0],
            'end_value': df['value'].iloc[-1],
            'count': len(df)
        }

    def detect_anomalies(self, values, threshold=2.0):
        """
        Detect anomalies using Z-score.
        
        Args:
            values (list): Numerical values.
            threshold (float): Z-score threshold (default 2.0).
            
        Returns:
            list of bool: True if anomaly, False otherwise.
        """
        y = np.array(values)
        if len(y) < 3: 
            return [False] * len(y) # Need reasonable sample size
            
        mean = np.mean(y)
        std = np.std(y)
        
        if std == 0:
            return [False] * len(y)
            
        z_scores = np.abs((y - mean) / std)
        return (z_scores > threshold).tolist()

    def get_patient_summary(self, patient_df, date_col='date', metric_col='value'):
        """
        Aggregates trends for a specific patient dataframe.
        """
        if metric_col not in patient_df.columns or date_col not in patient_df.columns:
            return {'error': 'Columns not found'}
            
        dates = patient_df[date_col]
        values = patient_df[metric_col]
        
        trend = self.analyze_trend(dates, values)
        anomalies = self.detect_anomalies(values)
        
        # Count anomalies
        trend['anomaly_count'] = sum(anomalies)
        
        return trend

if __name__ == "__main__":
    # Verification Run
    print("Testing TrendAnalyzer...")
    analyzer = TrendAnalyzer()
    
    # 1. Test Increasing Trend
    dates = pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01'])
    values = [5.5, 5.7, 5.9, 6.2] # Hba1c rising
    
    print(f"\nTest Data (Increasing): {values}")
    result = analyzer.analyze_trend(dates, values)
    print("Result:", result)
    
    # 2. Test Anomaly
    # With N=5, max Z-score is < 1.8. Need more points for Z > 2.0.
    values_with_outlier = [5.5, 5.6, 5.5, 5.4, 5.5, 5.6, 5.5, 5.5, 5.4, 5.6, 100.0, 5.5, 5.6, 5.5, 5.4]
    print(f"\nTest Data (Anomaly): {values_with_outlier}")
    anomalies = analyzer.detect_anomalies(values_with_outlier)
    print("Anomalies Detected:", anomalies)
    
    # Check expectation (index 10 is the outlier)
    if result['direction'] == 'Increasing' and anomalies[10] == True:
        print("\n✅ Verification PASSED")
    else:
        print("\n❌ Verification FAILED")
