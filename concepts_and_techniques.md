# Concepts, Techniques, and Formulas Used

This document outlines the statistical concepts, data analysis techniques, and formulas applied to generate the [Dataset Overview](file:///Users/ashishasharma/.gemini/antigravity/brain/062064f4-9be6-4d9f-83b9-5964b47ed058/dataset_overview.md) and the enhanced [EDA Visualizations](file:///Users/ashishasharma/Desktop/smarttech/healthcare-multimodal/healthcare_readmission_analysis.ipynb).

## 1. Descriptive Statistics

We employed descriptive statistics to summarize the main characteristics of the dataset.

*   **Frequency Distributions (Value Counts)**
    *   **Concept**: Determining how often each unique value appears in a categorical column.
    *   **Application**: Used for `readmitted` status, `race`, `gender`, and other categorical features.
    *   **Code**: `df['column'].value_counts()`

*   **Proportions (Percentages)**
    *   **Concept**: Normalizing counts to show the relative contribution of each category, making it easier to compare groups of different sizes.
    *   **Formula**:
        $$ \text{Percentage} = \left( \frac{\text{Count of Category}_i}{\text{Total Population}} \right) \times 100 $$
    *   **Application**: Showing that 53.9% of patients were not readmitted.
    *   **Code**: `df['column'].value_counts(normalize=True) * 100`

*   **Mean (Average)**
    *   **Concept**: The sum of all values divided by the number of values. In the context of binary targets (0 or 1), the mean represents the **proportion** or **rate** of the positive class.
    *   **Application**: Calculating the "30-Day Readmission Rate" for different groups (e.g., Age 70-80 vs. 20-30).
    *   **Code**: `df.groupby('feature')['target'].mean()`

## 2. Data Quality & Cleaning Concepts

*   **Missing Data Analysis**
    *   **Concept**: Identifying fields with absent or placeholder values (like `?`) that reduce the effective dataset size or quality.
    *   **Formula**:
        $$ \text{Missing \%} = \left( \frac{\text{Count of Missing Values}}{\text{Total Rows}} \right) \times 100 $$
    *   **Application**: Identified that `weight` (97% missing) and `max_glu_serum` (95% missing) were sparsely populated.

*   **Data Leakage Detection**
    *   **Concept**: Identifying features that contain information about the target variable that would not be available at the time of prediction (e.g., discharge codes indicating death).
    *   **Technique**: Analyzing feature definitions and checking for correlations with the outcome that logically happen *after* the prediction point.

*   **Class Imbalance**
    *   **Concept**: When one class in the target variable significantly outnumbers the others.
    *   **Formula**:
        $$ \text{Imbalance Ratio} = \frac{\text{Count of Minority Class}}{\text{Count of Majority Class}} $$
    *   **Application**: Noted an imbalance where `<30` days readmission (11%) is the minority compared to `NO` readmission (54%).

## 3. Visualization Techniques

*   **Univariate Analysis (Single Variable)**
    *   **Bar Charts**: Used to visualize the frequency distribution of categorical variables.
    *   **Dual-Axis Plotting**: We plotted **Counts** (absolute numbers) side-by-side with **Percentages** (relative contribution) to provide a complete picture.
    *   **Annotations**: Added data labels directly on top of bars to make the charts readable without referencing the y-axis.

*   **Bivariate Analysis (Two Variables)**
    *   **Grouped Aggregation**: Grouping data by a predictor variable (e.g., `age`) and calculating the statistics of the target variable (`readmission rate`).
    *   **Reference Lines**: Added a horizontal line representing the **Global Average** readmission rate. This allows for instant identification of which subgroups are "above average" or "below average" risk.
    *   **Formula for Plot**:
        $$ \text{Group Risk} = \text{Mean of Target for Group}_i $$

## 4. Key Metrics for Classification

Although not part of the visuals yet, these metrics define the modeling goal:

*   **Target Variable Definition**:
    We converted the multiclass target (`<30`, `>30`, `NO`) into a binary target for specific prediction goals:
    *   `readmit_30_days = 1` if `readmitted == '<30'`
    *   `readmit_30_days = 0` otherwise.
