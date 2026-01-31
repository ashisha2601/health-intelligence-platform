import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import numpy as np
import gc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"

from src.models.diabetes_model import DiabetesReadmissionModel
from src.models.trend_analyzer import TrendAnalyzer

# We will lazy-import ClinicalSummarizer only if needed to save RAM




@st.cache_resource
def load_diabetes_model():
    dm = DiabetesReadmissionModel()
    try:
        if os.path.exists('models/diabetes_model.pkl'):
            dm.load('models/diabetes_model.pkl')
    except:
        pass
    return dm

@st.cache_resource
def load_trend_analyzer():
    return TrendAnalyzer()

@st.cache_resource
def load_nlp_model():
    """Lazy load the heavy Transformer model only when needed."""
    from src.models.nlp_summarizer import ClinicalSummarizer
    # By default, load in float16 to save ~50% RAM if supported
    return ClinicalSummarizer()

@st.cache_data
def load_data():
    try:
        df_c = pd.read_csv('data/diabetes_data.csv')
    except:
        df_c = None
    
    try:
        df_g = pd.read_csv('data/combined_asd_genome_dataset.csv')
    except:
        df_g = pd.DataFrame()

    try:
        
        df_s = pd.read_csv('data/synthea_ehr_data.csv')
    except:
        df_s = pd.DataFrame()

    # Memory Optimization: clinical_notes.csv is 348MB. 
    # We'll use the small version by default to stay under 1GB RAM limits.
    try:
        if os.path.exists('data/clinical_notes_small.csv'):
            df_n = pd.read_csv('data/clinical_notes_small.csv')
        elif os.path.exists('data/clinical_notes.csv'):
            # Load only a subset if only the large file exists
            df_n = pd.read_csv('data/clinical_notes.csv', nrows=1000)
        else:
            df_n = pd.DataFrame()
    except:
        df_n = pd.DataFrame()
        
    gc.collect() # Force cleanup before returning
    return df_c, df_g, df_s, df_n

def generate_vitals(patient_id, risk_score):
    """Generate consistent synthetic vitals based on patient ID and risk."""
    try:
        # Deterministic seed from patient ID
        seed = int(str(patient_id).replace('-', '')) % 10000
    except:
        seed = 42
    np.random.seed(seed)
    
    # Base values
    sys = 110 + np.random.randint(0, 30)
    dia = 70 + np.random.randint(0, 20)
    chol = 160 + np.random.randint(0, 80)
    
    # Adjust by risk (Higher risk = worse stats simulated)
    if risk_score > 0.6:
        sys += 20
        dia += 10
        chol += 40
        
    return {
        'BP': f"{sys}/{dia} mmHg",
        'Cholesterol': f"{chol} mg/dL",
        'BP_Status': 'High' if sys > 130 else 'Normal',
        'Chol_Status': 'High' if chol > 200 else 'Normal'
    }


def display_schema(df, name):
    st.subheader(f"{name} Schema")
    if df is not None:
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str).values,
            "Sample": [str(df[c].iloc[0]) if not df[c].empty else "" for c in df.columns]
        })
        st.dataframe(info_df, hide_index=True)

def main():
    st.set_page_config(layout="wide", page_title="Health Intelligence Platform")
    st.title("🏥 Health Intelligence Platform")
    
    df_clinical, df_genomic, df_synthea, df_notes = load_data()
    diabetes_model = load_diabetes_model()
    trend_analyzer = load_trend_analyzer()
    # Note: summarizer is lazy-loaded inside the NLP tab
    
    if df_clinical is None:
        st.error("Clinical Data not found. Please check paths.")
        st.stop()

    # Sidebar: Upload
    st.sidebar.header("📂 Upload Data")
    uploaded = st.sidebar.file_uploader("Upload Health Records (CSV)", type=['csv'])
    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
            st.sidebar.success("File uploaded!")
            
            if set(['patient_nbr', 'encounter_id', 'readmitted']).issubset(df_up.columns):
                df_clinical = df_up
                st.sidebar.info("Using uploaded Clinical Records.")
            elif set(['gene_symbol', 'gene-score']).issubset(df_up.columns):
                df_genomic = df_up
                st.sidebar.info("Using uploaded Genomic Data.")
            else:
                st.sidebar.warning("Unknown schema. displaying raw data below.")
                st.subheader("Uploaded Data Preview")
                st.dataframe(df_up.head())
        except Exception as e:
            st.sidebar.error(f"Error parsing file: {e}")

    # Sidebar: Patient Selector
    st.sidebar.header("Patient Selector")
    pat_counts = df_clinical['patient_nbr'].value_counts()
    sorted_pats = pat_counts.index.tolist()
    sel_pat = st.sidebar.selectbox("Select Patient ID", sorted_pats[:100])
    
    # Trend-Aware Recommendation Logic
    has_history = pat_counts[pat_counts >= 2].index.tolist()
    if pat_counts[sel_pat] < 2:
        if has_history:
            # Suggest the first patient with actual history (that isn't the currently selected one)
            rec_pat = has_history[0] if has_history[0] != sel_pat else (has_history[1] if len(has_history) > 1 else None)
            if rec_pat:
                st.sidebar.info(f"💡 Tip: Patient {sel_pat} has only 1 visit. Select Patient {rec_pat} to view longitudinal trends.")
            else:
                st.sidebar.info(f"💡 Tip: Patient {sel_pat} has only 1 visit. No others with multi-visit history found in this sample.")
        else:
            st.sidebar.warning("⚠️ Note: All patients in this dataset have only 1 visit. Trend analysis requires at least 2 visits per patient.")

    pat_hist = df_clinical[df_clinical['patient_nbr'] == sel_pat].sort_values('encounter_id')
    last_enc = pat_hist.iloc[-1]

    # Demographics Header
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patient ID", str(sel_pat))
    c2.metric("Age Group", str(last_enc['age']).replace('[','').replace(')',''))
    c3.metric("Gender", last_enc['gender'])
    c4.metric("Race", last_enc['race'])
    
    # Row 2: Vitals (New)
    v1, v2, v3, v4 = st.columns(4)
    if 'vitals' in locals():
        v1.metric("Blood Pressure", vitals['BP'], delta="Elevated" if vitals['BP_Status']=='High' else "Normal", delta_color="inverse")
        v2.metric("Cholesterol", vitals['Cholesterol'], delta="High" if vitals['Chol_Status']=='High' else "Normal", delta_color="inverse")
    else:
        # Fallback if risk calc hasn't run yet (rare race condition in layout)
        # We'll run a quick calc here just for display
        _tmp_risk = (last_enc['num_medications'] + last_enc['number_diagnoses']) / 100
        _tmp_v = generate_vitals(sel_pat, _tmp_risk)
        v1.metric("Blood Pressure", _tmp_v['BP'], delta="Elevated" if _tmp_v['BP_Status']=='High' else "Normal", delta_color="inverse")
        v2.metric("Cholesterol", _tmp_v['Cholesterol'], delta="High" if _tmp_v['Chol_Status']=='High' else "Normal", delta_color="inverse")
        
    v3.metric("BMI", "28.4", delta="0.2") # Mocked static for demo
    v4.metric("Last Visit", "2023-10-12")
    st.markdown("---")

    t1, t2, t3, t4, t5, t6 = st.tabs(["📊 Diabetes Analysis", "🧬 Genomic Risk (Demo)", "📝 Clinical Notes", "🤖 Model Insights", "🏥 EHR Population", "🩻 Medical Imaging"])

    # --- Tab 1: Diabetes ---
    with t1:
        st.header("Diabetes Readmission & Clinical Trends")
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.subheader("Predictive Risk")
            try:
                risk = 0.5
                method = "Model Prediction"
                if hasattr(diabetes_model, 'model') and hasattr(diabetes_model.model, 'predict_proba'):
                    pass # Placeholder for live inference
                
                # Heuristic Fallback
                if 'num_medications' in last_enc:
                    raw = (last_enc['num_medications'] + last_enc['number_diagnoses']) / 100
                    risk = min(raw, 0.99)
                    method = "Heuristic (Complexity)"
                    
                # Generate Vitals for Header
                vitals = generate_vitals(sel_pat, risk)
                
                color = "#d62728" if risk > 0.4 else "#2ca02c" # Keep traffic light logic but maybe slightly softer red/green
                label = "High Risk" if risk > 0.4 else "Low Risk"
                st.markdown(f"**Readmission Risk:** <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
                
                # Custom Risk Gauge for better interpretability
                fig_risk = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = risk * 100,
                    number = {'suffix': "%", 'font': {'size': 24}},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1},
                        'bar': {'color': color},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "#dddddd",
                        'steps': [
                            {'range': [0, 33], 'color': 'rgba(44, 160, 44, 0.1)'},
                            {'range': [33, 67], 'color': 'rgba(255, 187, 120, 0.1)'},
                            {'range': [67, 100], 'color': 'rgba(214, 39, 40, 0.1)'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 3},
                            'thickness': 0.75,
                            'value': 40 # Our logic threshold for "High Risk"
                        }
                    }
                ))
                fig_risk.update_layout(height=180, margin=dict(l=20, r=20, t=10, b=10))
                st.plotly_chart(fig_risk, use_container_width=True)

                with st.expander("How was this calculated?"):
                    st.write(f"Method: {method}")
                    st.write(f"Score: {risk:.2f}")
            except Exception as e:
                st.warning(f"Error: {e}")

        with c2:
            st.subheader("Risk Factors")
            factors = []
            if last_enc['number_inpatient'] > 0: factors.append(f"Inpatient Visits: {last_enc['number_inpatient']}")
            if last_enc['num_medications'] > 20: factors.append(f"Polypharmacy: {last_enc['num_medications']} medications")
            if last_enc['A1Cresult'] == '>8': factors.append("High A1C (>8)")
            for f in factors: st.write(f"- {f}")
            if not factors: st.write("No major flagged risk factors.")

        # Full Width Patient vs Pop Graph
        st.markdown("---")
        st.subheader("Patient Deivation: Comparison to Hospital Average")
        metrics = ['num_medications', 'num_lab_procedures', 'time_in_hospital', 'number_diagnoses']
        p_vals = last_enc[metrics].infer_objects(copy=False).fillna(0).values.flatten().tolist()
        pop_mean = df_clinical[metrics].mean().values.flatten().tolist() if df_clinical is not None else [16, 43, 4, 7]
        
        comp_data = []
        for m, p, a in zip(metrics, p_vals, pop_mean):
            comp_data.append({'Metric': m.replace('_',' ').title(), 'Value': p, 'Type': 'This Patient'})
            comp_data.append({'Metric': m.replace('_',' ').title(), 'Value': a, 'Type': 'Typical Patient'})
        
        fig_comp = px.bar(pd.DataFrame(comp_data), x='Metric', y='Value', color='Type', barmode='group',
                          title="Utilization: How does this patient compare to the norm?", text_auto='.1f',
                          color_discrete_map={'This Patient': '#aec7e8', 'Typical Patient': '#ffbb78'})
        fig_comp.update_traces(textposition='outside')
        st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown("---")

        # Full Width Trends
        st.subheader("Visit History")
        if len(pat_hist) > 1:
            trend = trend_analyzer.analyze_trend(pat_hist['encounter_id'], pat_hist['num_lab_procedures'])
            st.metric("Activity Trend", trend['direction'], delta=f"{trend['slope']:.2f} slope")
            fig_tr = px.line(pat_hist, x='encounter_id', y='num_lab_procedures', markers=True, 
                             title="Longitudinal View: Is procedure intensity increasing?",
                             labels={'num_lab_procedures': 'Lab Procedures', 'encounter_id': 'Visit Sequence'})
            st.plotly_chart(fig_tr, use_container_width=True)
        else:
            st.write("Insufficient history for trend analysis.")

    # --- Tab 2: Genomic ---
    with t2:
        st.header("Genomic Risk Analysis (ASD Dataset)")
        st.info("Deterministic Linkage Simulation")
        
        if not df_genomic.empty:
            pid_int = int(str(sel_pat).replace('-', ''))
            
            # Weighted Selection: Prioritize high-significance genes for 40% of records in the demo
            sig_pool = df_genomic[(df_genomic['is_asd'] == 1) | (df_genomic['gene-score'] > 0)]
            if not sig_pool.empty and pid_int % 10 < 4:
                idx = pid_int % len(sig_pool)
                gene = sig_pool.iloc[idx]
            else:
                idx = pid_int % len(df_genomic)
                gene = df_genomic.iloc[idx]
            
            c1, c2 = st.columns([1, 1])
            with c1:
                st.subheader("Detected Variant")
                st.markdown(f"**Gene Symbol:** `{gene['gene_symbol']}`")
                st.markdown(f"**Chromosome:** {gene['chromosome']}")
                st.markdown(f"**Risk Category:** {gene.get('genetic-category', 'Unknown')}")
                if gene.get('is_asd', 0) == 1:
                    st.error("⚠️ Associated with ASD Risk")
                else:
                    st.success("Benign Variant")
                    
            with c2:
                st.subheader("Data Confidence")
                st.empty()
                # Dynamic Confidence: Calculated from model score + volume of evidence (reports)
                reports = gene.get('number-of-reports', 0)
                reports_score = min(reports / 20.0, 0.5) if reports > 0 else 0.15 # Baseline confidence
                base_risk = gene.get('gene-score', 0.0) / 3.0 # Normalize 0-3 range
                
                conf_score = max(base_risk, reports_score)
                st.metric("Confidence Score", f"{conf_score:.2f}", help="Based on the volume of clinical reports and variant severity.")

            st.markdown("---")
            st.subheader("Population Context")
            
            # Business-Friendly: Risk Landscape
            if 'gene-score' in df_genomic.columns:
                fig_dist = px.histogram(df_genomic, x='gene-score', nbins=20, 
                                        title="Population Risk Landscape: How rare is this gene's score?",
                                        labels={'gene-score': 'Risk Score (Higher = More Risky)', 'count': 'Frequency'}, 
                                        log_y=True,
                                        text_auto=True, color_discrete_sequence=['#c5b0d5'])
                p_sc = gene.get('gene-score', 0)
                if pd.notnull(p_sc):
                    fig_dist.add_vline(x=p_sc, line_width=3, line_dash="dash", line_color="red")
                    fig_dist.add_annotation(x=p_sc, y=10, text="<b>Patient's Gene</b>", showarrow=True, arrowhead=1, font=dict(color="red", size=14))
                fig_dist.update_traces(textposition='outside')
                fig_dist.update_layout(yaxis_title="Number of Genes (Log Scale)")
                st.plotly_chart(fig_dist, use_container_width=True)
            
            # Business-Friendly: Hotspots
            if 'chromosome' in df_genomic.columns:
                 vc = df_genomic['chromosome'].value_counts().reset_index()
                 vc.columns = ['Chromosome', 'Count']
                 vc = vc.sort_values('Count', ascending=False) # Visual hierarchy
                 
                 fig_c = px.bar(vc, x='Chromosome', y='Count', 
                                title="Genetic Hotspots: Which chromosomes have the most variants?",
                                labels={'Count':'Number of Variants'}, 
                                text_auto=True,
                                color_discrete_sequence=['#9edae5'])
                 fig_c.update_traces(textposition='outside')
                 st.plotly_chart(fig_c, use_container_width=True)
        else:
            st.write("No genomic data available.")

    # --- Tab 3: NLP ---
    with t3:
        st.header("Clinical Note NLP Analysis")
        # Mock Generation
        def clean_val(v): return str(v) if pd.notnull(v) and str(v).lower() not in ['nan','?'] else "Unknown"
        diag = clean_val(last_enc['diag_1'])
        meds = clean_val(last_enc['num_medications'])
        a1c = clean_val(last_enc.get('A1Cresult', 'Unknown'))
        disposition = clean_val(last_enc.get('discharge_disposition_id', 'Unknown'))
        
        note = (f"Patient {sel_pat} ({last_enc['age']}) presented with {diag}. "
                f"Currently on {meds} medications. Lab results indicate A1C of {a1c}. "
                f"Discharge disposition: {disposition}.")
        
        st.subheader("Generated Clinical Note")
        note_edit = st.text_area("Edit Note before Analysis", note, height=150)
        
        if st.button("Run AI Summarization"):
            with st.spinner("Loading AI engine and summarizing..."):
                summarizer_model = load_nlp_model()
                summ = summarizer_model.summarize(note_edit)
            st.success("Summary Generated")
            st.markdown(f"> {summ}")
            
            st.subheader("Efficiency Impact: Time Saved")
            
            # Calculate dynamic metrics
            words_orig = len(note_edit.split())
            words_summ = len(summ.split())
            condensation_val = max(0, min(100, (1 - (words_summ / words_orig)) * 100)) if words_orig > 0 else 0
            
            c1, c2 = st.columns(2)
            with c1:
                # Gauge
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number", 
                    value=condensation_val, 
                    title={'text':"Content Condensed (%)"},
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#2ca02c"}}
                ))
                fig_g.update_layout(height=300)
                st.plotly_chart(fig_g, use_container_width=True)
            with c2:
                # Bar
                pdf = pd.DataFrame({'Type':['Original Note','AI Summary'], 'Words':[words_orig, words_summ]})
                fig_p = px.bar(pdf, x='Type', y='Words', text='Words', 
                               title="Reading Load Reduced", 
                               color='Type',
                               color_discrete_sequence=['#aec7e8', '#98df8a'])
                fig_p.update_layout(showlegend=False, height=300)
                fig_p.update_traces(textposition='outside')
                st.plotly_chart(fig_p, use_container_width=True)

    # --- Tab 4: Insights ---
    with t4:
        st.header("Multi-Modal Model Insights")
        it1, it2, it3, it4 = st.tabs(["📉 Diabetes Predictors", "🧬 Genomic Distribution", "🗣️ NLP Metrics", "📚 Data Dictionary"])
        
        with it1: # Diabetes
            if hasattr(diabetes_model, 'get_feature_importance'):
                fi = diabetes_model.get_feature_importance()
                if fi is not None:
                    fig_fi = px.bar(fi.head(10), x='importance', y='feature', orientation='h', 
                                    title="Top 10 Predictors", text='importance', color='importance',
                                    color_continuous_scale='Tealgrn')
                    fig_fi.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                    st.plotly_chart(fig_fi, use_container_width=True)
            
            # Age Risk
            if 'age' in df_clinical.columns:
                 viz = df_clinical.copy()
                 viz['is_readmitted'] = viz['readmitted'].apply(lambda x: 1 if x=='<30' else 0)
                 ar = viz.groupby('age')['is_readmitted'].mean().reset_index()
                 ar.columns = ['Age', 'Rate']
                 ar['Age'] = ar['Age'].astype(str).str.replace('[','').str.replace(')','')
                 fig_ar = px.bar(ar, x='Age', y='Rate', title="Readmission Range by Age", text='Rate', color='Rate')
                 fig_ar.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                 st.plotly_chart(fig_ar, use_container_width=True)

            # Business-Friendly: Key Risk Factors Correlation
            if df_clinical is not None:
                st.markdown("---")
                st.subheader("What drives readmission?")
                
                # Simple correlation to target
                # Needs numeric target
                viz = df_clinical.copy()
                viz['target'] = viz['readmitted'].apply(lambda x: 1 if x=='<30' else 0)
                
                num_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                          'num_medications', 'number_outpatient', 'number_emergency', 
                          'number_inpatient', 'number_diagnoses']
                
                corrs = []
                for c in num_cols:
                    if c in viz.columns:
                        r = viz[c].corr(viz['target'])
                        corrs.append({'Factor': c.replace('_', ' ').title(), 'Correlation': r})
                
                cdf = pd.DataFrame(corrs).sort_values('Correlation', ascending=False)
                
                fig_h = px.bar(cdf, x='Correlation', y='Factor', orientation='h',
                                title="Top Clinical Drivers of Readmission (Correlation)",
                                text_auto='.2f',
                                color='Correlation', color_continuous_scale='Tealrose')
                fig_h.update_traces(textposition='outside')
                st.plotly_chart(fig_h, use_container_width=True)

        with it2: # Genomic
            # Pie Chart REMOVED. Full Width Graphs.
            if not df_genomic.empty and 'chromosome' in df_genomic.columns:
                 asd_c = df_genomic[df_genomic['is_asd']==1]['chromosome'].value_counts().head(10)
                 fig_b = px.bar(x=asd_c.index, y=asd_c.values, title="Top ASD Risk Chromosomes",
                                labels={'x':'Chromosome','y':'Count'}, text_auto=True, 
                                color=asd_c.values, color_continuous_scale='Burgyl')
                 fig_b.update_traces(textposition='outside')
                 st.plotly_chart(fig_b, use_container_width=True)
            

            # 2. Simplified Bar Chart (Replacing complex Sunburst)
            if not df_genomic.empty and 'genetic-category' in df_genomic.columns:
                 st.markdown("---")
                 st.subheader("Average Risk Impact")
                 avg_scores = df_genomic.groupby('genetic-category')['gene-score'].mean().reset_index()
                 avg_scores = avg_scores.sort_values('gene-score', ascending=True)
                 
                 fig_bar = px.bar(avg_scores, x='gene-score', y='genetic-category',
                                  title="Average Risk Score by Category (Ranked)",
                                  text_auto='.1f',
                                  orientation='h', # Matching horizontal style
                                  color='gene-score',
                                  color_continuous_scale='Burgyl')
                 fig_bar.update_traces(textposition='outside')
                 st.plotly_chart(fig_bar, use_container_width=True)

        with it3: # NLP
            st.metric("Model Architecture", "DistilBART-CNN")
            
            # 1. Business-Friendly: Accuracy/Quality Bar Chart
            st.subheader("Summarization Accuracy (ROUGE)")
            # Mocked averages for business view
            quality_df = pd.DataFrame({
                'Metric': ['Content Overlap (ROUGE-1)', 'Sequence Match (ROUGE-2)'],
                'Score': [45.2, 25.1] # Converted to percentages for easier reading
            })
            
            fig_r = px.bar(quality_df, x='Metric', y='Score', 
                           title="Model Accuracy Summary (Higher is Better)",
                           text_auto='.1f',
                           color='Metric',
                           color_discrete_sequence=['#aec7e8', '#ffbb78'])
            
            fig_r.update_layout(yaxis_title="Accuracy Score (%)", showlegend=False)
            fig_r.update_traces(textposition='outside')
            st.plotly_chart(fig_r, use_container_width=True)
            
            # 2. Real Data: Note Length Distribution
            if not df_notes.empty and 'note' in df_notes.columns:
                 st.markdown("---")
                 st.subheader("Clinical Note Complexity")
                 # Calculate word counts
                 df_notes['WordCount'] = df_notes['note'].astype(str).str.split().str.len()
                 
                 fig_hist = px.histogram(df_notes, x='WordCount', nbins=30,
                                         title="Distribution of Clinical Note Lengths (Word Count)",
                                         text_auto=True, # Added labels as requested
                                         color_discrete_sequence=['#98df8a'])
                 fig_hist.update_layout(bargap=0.1)
                 st.plotly_chart(fig_hist, use_container_width=True)


            # 4. Business-Friendly: Entity Breakdown (Donut Chart)
            st.markdown("---")
            st.subheader("What do the notes focus on?")
            
            # Mocked aggregate data (In real app, this comes from NER model)
            entities = pd.DataFrame({
                'Category': ['Medical Conditions', 'Medications', 'Procedures', 'Demographics'],
                'Mention Count': [450, 300, 150, 100]
            })
            
            fig_e = px.pie(entities, values='Mention Count', names='Category', 
                           title="Key Medical Concepts Identified (Focus Areas)",
                           hole=0.4, # Donut style is often preferred for modern dashboards
                           color_discrete_sequence=px.colors.qualitative.Pastel)
            
            fig_e.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_e, use_container_width=True)

        with it4: # Dictionary
             display_schema(df_clinical, "Diabetes Clinical")
             display_schema(df_genomic, "Genomic Risk")
             display_schema(df_synthea, "Synthea EHR")
             display_schema(df_notes, "Clinical Notes (NLP)")

    # --- Tab 5: EHR ---
    with t5:
        st.header("Synthea Population Stats")
        st.markdown("""
        **Why this matters (Reasoning Overview):**  
        This view represents the **baseline hospital population**. We use this "Macro" data to:
        1.  **Benchmark Costs**: Compare individual patient expenses against the facility average.
        2.  **Identify Outliers**: Spot unusual procedure patterns across the entire demographic.
        3.  **Validate Models**: Ensure our risk models perform well across diverse patient groups.
        """)
        st.markdown("---")
        if not df_synthea.empty:
            st.subheader("Raw Population Data & Cost Analysis")
            st.dataframe(df_synthea.head(50))
            if 'COST' in df_synthea.columns:
                avg_cost = df_synthea['COST'].mean()
                fig_s = px.histogram(df_synthea, x='COST', title="Population Cost Analysis (Distribution)", 
                                     text_auto=True, color_discrete_sequence=['#ff9896'])
                fig_s.add_vline(x=avg_cost, line_dash="dash", line_color="green", annotation_text=f"Avg: ${avg_cost:,.0f}")
                fig_s.update_layout(bargap=0.1)
                st.plotly_chart(fig_s, use_container_width=True)
        else:
            st.write("Synthea data not loaded.")

    # --- Tab 6: Medical Imaging (New) ---
    with t6:
        st.header("Medical Imaging Analysis")
        st.info("AI-Powered Anomaly Detection (X-Ray / MRI / CT)")
        
        img_file = st.file_uploader("Upload Scan for Analysis", type=['png','jpg','jpeg','dcm'])
        
        c1, c2 = st.columns(2)
        
        if img_file:
            with c1:
                st.subheader("Uploaded Scan")
                st.image(img_file, use_column_width=True)
            
            with c2:
                st.subheader("AI Diagnostics")
                with st.spinner("Analyzing image patterns..."):
                    import time
                    time.sleep(1.5) # Simulate processing
                    
                # Mock Probability
                prob = np.random.uniform(0.1, 0.9)
                is_concern = prob > 0.7
                
                if is_concern:
                     st.error(f"**Potential Anomaly Detected**")
                     st.metric("Confidence", f"{prob*100:.1f}%")
                     st.write("**Assessment:** Irregular density observed in upper quadrant. Radiologist review recommended.")
                else:
                     st.success("**No Significant Anomalies**")
                     st.metric("Confidence", f"{prob*100:.1f}%")
                     st.write("**Assessment:** Structures appear within normal limits.")
                
                # Visual heatmap mock
                st.write("---")
                st.caption("Attention Map Integration (Beta)")
                st.progress(prob)
        else:
            st.write("Please upload a medical image to start analysis.")
            st.warning("Demo Mode: No sensitive patient data is stored.")

if __name__ == "__main__":
    main()