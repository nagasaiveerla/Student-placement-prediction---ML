import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.model_training import EnsembleClassifier
from src.data_preprocessing import DataPreprocessor

def create_streamlit_app():
    """
    Streamlit web application for student placement prediction
    """
    
    st.set_page_config(
        page_title="Student Placement Predictor",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🎓 Student Placement Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("### Predict student placement probability using advanced ML")
    
    # Load model and preprocessor
    try:
        model = EnsembleClassifier.load_model()
        preprocessor = DataPreprocessor.load_preprocessor()
        st.success("✅ Model loaded successfully!")
    except FileNotFoundError:
        st.error("❌ Model files not found. Please run training first!")
        st.code("python train_model.py")
        st.stop()
    
    # Sidebar - Input Form
    st.sidebar.header("📝 Enter Student Details")
    st.sidebar.markdown("---")
    
    with st.sidebar:
        st.subheader("Academic Information")
        iq = st.number_input("IQ Score", min_value=70, max_value=150, value=100, help="Student's IQ score")
        prev_sem = st.number_input("Previous Semester Result", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
        academic_perf = st.slider("Academic Performance (1-10)", 1, 10, 7)
        
        st.subheader("Experience & Skills")
        internship = st.selectbox("Internship Experience", ["Yes", "No"])
        extra_curr = st.slider("Extra Curricular Activities (0-10)", 0, 10, 5)
        comm_skill = st.slider("Communication Skills (1-10)", 1, 10, 7)
        projects = st.slider("Projects Completed (0-10)", 0, 10, 3)
        
        st.markdown("---")
        predict_btn = st.button("🔮 Predict Placement", type="primary", use_container_width=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if predict_btn:
            # Prepare features
            features = {
                'IQ': iq,
                'Prev_Sem_Result': prev_sem,
                'CGPA': cgpa,
                'Academic_Perform': academic_perf,
                'Internship_Experience': internship,
                'Extra_Curricular': extra_curr,
                'Communication_Skill': comm_skill,
                'Projects_Completed': projects
            }
            
            # Transform features
            processed_features = preprocessor.transform_features(features)
            
            # Create DataFrame for prediction
            feature_df = pd.DataFrame([processed_features])
            
            # Make prediction
            prediction = model.predict(feature_df)[0]
            prediction_proba = model.predict_proba(feature_df)[0]
            
            # Display results
            st.markdown("## 🎯 Prediction Results")
            
            if prediction == 1:
                st.success("🎉 **STUDENT WILL BE PLACED!**")
                confidence = prediction_proba[1] * 100
                result_color = "green"
            else:
                st.error("❌ **STUDENT MAY NOT BE PLACED**")
                confidence = prediction_proba[0] * 100
                result_color = "red"
            
            # Confidence metric
            col_conf1, col_conf2, col_conf3 = st.columns(3)
            with col_conf1:
                st.metric("Confidence", f"{confidence:.1f}%")
            with col_conf2:
                st.metric("Placement Probability", f"{prediction_proba[1]*100:.1f}%")
            with col_conf3:
                st.metric("Risk Level", "Low" if confidence > 80 else "Medium" if confidence > 60 else "High")
            
            # Probability visualization
            st.subheader("📊 Prediction Probabilities")
            
            prob_data = {
                'Outcome': ['Not Placed', 'Placed'],
                'Probability': [prediction_proba[0], prediction_proba[1]],
                'Color': ['red', 'green']
            }
            
            fig = px.bar(
                prob_data, 
                x='Outcome', 
                y='Probability',
                color='Color',
                color_discrete_map={'red': '#ff4444', 'green': '#44ff44'},
                title="Placement Probability Distribution"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature contribution (if available)
            if hasattr(model.model, 'feature_importances_'):
                st.subheader("🎯 Feature Impact Analysis")
                
                importance_data = {
                    'Feature': model.feature_names,
                    'Importance': model.model.feature_importances_,
                    'Your_Value': [features[f] for f in model.feature_names]
                }
                
                importance_df = pd.DataFrame(importance_data).sort_values('Importance', ascending=True)
                
                fig2 = px.bar(
                    importance_df.tail(8), 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title="Top Features Influencing Placement Decision"
                )
                st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.subheader("📈 Model Information")
        
        # Model stats
        if hasattr(model, 'cv_results') and model.cv_results:
            st.markdown("**Cross-Validation Results:**")
            results = model.cv_results
            
            for metric, values in results.items():
                st.metric(
                    metric.replace('_', ' ').title(),
                    f"{values['mean']:.3f}",
                    f"±{values['std']:.3f}"
                )
        
        # Sample distribution
        st.subheader("📊 Training Data Stats")
        st.info("Model trained on student placement data with balanced features for optimal performance.")
        
        # Tips for improvement
        if predict_btn and prediction == 0:
            st.subheader("💡 Improvement Suggestions")
            suggestions = []
            
            if cgpa < 7.0:
                suggestions.append("📚 Focus on improving CGPA")
            if comm_skill < 7:
                suggestions.append("🗣️ Enhance communication skills")
            if projects < 3:
                suggestions.append("🔨 Work on more projects")
            if internship == "No":
                suggestions.append("💼 Gain internship experience")
            if extra_curr < 5:
                suggestions.append("🏃 Participate in extra-curricular activities")
            
            for suggestion in suggestions:
                st.write(f"• {suggestion}")

if __name__ == "__main__":
    create_streamlit_app()