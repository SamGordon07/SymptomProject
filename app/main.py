import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import sqlite3
import shap
import matplotlib.pyplot as plt

# Add src to path so we can import predict.py
sys.path.append('src')
from predict import load_models, predict_diseases

# ── Page configuration ─────────────────────────────────────────────
st.set_page_config(
    page_title='Symptom Checker',
    page_icon='🩺',
    layout='centered'
)


@st.cache_resource
def get_models():
    feature_cols, disease_models, confidence_tiers = load_models(model_dir='models')
    return feature_cols, disease_models, confidence_tiers

feature_cols, disease_models, confidence_tiers = get_models()


def log_query(symptoms, age, gender, blood_pressure, cholesterol, predictions):
    try:
        conn = sqlite3.connect('data/symptoms.db')
        cursor = conn.cursor()

        # Create table if it doesn't exist yet
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_queries (
                query_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                fever INTEGER,
                cough INTEGER,
                fatigue INTEGER,
                difficulty_breathing INTEGER,
                age INTEGER,
                gender TEXT,
                blood_pressure TEXT,
                cholesterol_level TEXT,
                top_prediction TEXT,
                top_confidence REAL
            )
        """)

        cursor.execute("""
            INSERT INTO user_queries (
                timestamp, fever, cough, fatigue,
                difficulty_breathing, age, gender,
                blood_pressure, cholesterol_level,
                top_prediction, top_confidence
            )
            VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symptoms.get('Fever', 0),
            symptoms.get('Cough', 0),
            symptoms.get('Fatigue', 0),
            symptoms.get('Difficulty Breathing', 0),
            age, gender, blood_pressure, cholesterol,
            predictions[0][0],   # top disease name
            predictions[0][1]    # top confidence score
        ))

        conn.commit()
        conn.close()
    except Exception as e:
        # Don't crash the app if logging fails
        st.warning(f'Query logging failed: {e}')
        
        
st.title('🩺 Symptom Checker')
st.markdown(
    'Enter your symptoms and patient profile below to see possible conditions. '
    'This tool is for informational purposes only and is **not a substitute '
    'for professional medical advice.**'
)
st.divider()

st.subheader('Symptoms')
st.caption('Select all symptoms that currently apply to you.')

col1, col2 = st.columns(2)

with col1:
    fever = st.checkbox('Fever')
    cough = st.checkbox('Cough')

with col2:
    fatigue = st.checkbox('Fatigue')
    difficulty_breathing = st.checkbox('Difficulty Breathing')
    

st.divider()
st.subheader('Patient Profile')
st.caption('This information helps narrow down predictions.')

col3, col4 = st.columns(2)

with col3:
    age = st.slider('Age', min_value=19, max_value=90, value=35)
    gender = st.radio('Gender', options=['Female', 'Male'], horizontal=True)

with col4:
    blood_pressure = st.selectbox(
        'Blood Pressure',
        options=['Low', 'Normal', 'High']
    )
    cholesterol = st.selectbox(
        'Cholesterol Level',
        options=['Low', 'Normal', 'High']
    )
    
    
def encode_inputs(fever, cough, fatigue, difficulty_breathing,
                  age, gender, blood_pressure, cholesterol):

    bp_map = {'Low': 0, 'Normal': 1, 'High': 2}
    chol_map = {'Low': 0, 'Normal': 1, 'High': 2}

    return {
        'Fever': int(fever),
        'Cough': int(cough),
        'Fatigue': int(fatigue),
        'Difficulty Breathing': int(difficulty_breathing),
        'Age': age,
        'Gender': 1 if gender == 'Male' else 0,
        'Blood Pressure': bp_map[blood_pressure],
        'Cholesterol Level': chol_map[cholesterol]
    }
    
    


# --- Main Logic --- #

st.divider()

if st.button('Check Symptoms', type='primary', use_container_width=True):

    # Encode the inputs
    user_features = encode_inputs(
        fever, cough, fatigue, difficulty_breathing,
        age, gender, blood_pressure, cholesterol
    )

    # Check at least one symptom is selected
    symptom_values = [fever, cough, fatigue, difficulty_breathing]
    if not any(symptom_values):
        st.warning('Please select at least one symptom before checking.')
        st.stop()

    # Run predictions
    with st.spinner('Analysing symptoms...'):
        predictions = predict_diseases(
            user_features,
            disease_models,
            feature_cols,
            top_n=5
        )

    # Log to database
    symptoms_dict = {
        'Fever': int(fever),
        'Cough': int(cough),
        'Fatigue': int(fatigue),
        'Difficulty Breathing': int(difficulty_breathing)
    }
    log_query(symptoms_dict, age, gender, blood_pressure, cholesterol, predictions)

    # ── Display results ────────────────────────────────────────────
    st.subheader('Possible Conditions')
    st.caption(
        f'Based on your inputs, here are the top {len(predictions)} '
        f'possible conditions ranked by likelihood.'
    )

    # Results as a bar chart
    diseases = [p[0] for p in predictions]
    confidences = [p[1] for p in predictions]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(diseases[::-1], confidences[::-1], color='steelblue')
    ax.set_xlabel('Confidence (%)')
    ax.set_title('Top predicted conditions')
    ax.bar_label(bars, fmt='%.1f%%', padding=3)
    ax.set_xlim(0, max(confidences) * 1.2)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Results as a table with confidence tier warnings
    st.subheader('Breakdown')
    for i, (disease, confidence) in enumerate(predictions):
        tier = confidence_tiers.get(disease, 'Low')
        medal = ['🥇', '🥈', '🥉', '4.', '5.'][i]

        col_a, col_b, col_c = st.columns([3, 2, 2])
        with col_a:
            st.write(f'{medal} **{disease}**')
        with col_b:
            st.write(f'{confidence}% confidence')
        with col_c:
            if tier == 'High':
                st.success('High data quality')
            elif tier == 'Medium':
                st.info('Medium data quality')
            else:
                st.warning('Limited training data')
                
    
    # ── SHAP explanation ───────────────────────────────────────────
    top_disease = predictions[0][0]

    if top_disease in disease_models:
        st.divider()
        st.subheader(f'Why {top_disease}?')
        st.caption(
           'This chart shows which of your inputs most influenced '
           'the top prediction. Blue bars pushed toward this diagnosis, '
           'red bars pushed away from it.'
        )

        try:
            top_model = disease_models[top_disease]
            explainer = shap.TreeExplainer(top_model)
            input_df = pd.DataFrame([user_features])[feature_cols]
            shap_values = explainer.shap_values(input_df)

            # Shape is (1, 8, 2) — sample, feature, class
            # Take class 1 (disease present) for the first sample
            sv_array = np.array(shap_values)
            sv = sv_array[0, :, 1]  # first sample, all features, class 1

            # Build the chart
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            colors = ['steelblue' if v >= 0 else 'coral' for v in sv]
            bars = ax2.barh(feature_cols, sv, color=colors)
            ax2.axvline(x=0, color='black', linewidth=0.8)
            ax2.bar_label(bars, fmt='%.3f', padding=3)
            ax2.set_xlabel('SHAP value — impact on prediction')
            ax2.set_title(f'What drove the {top_disease} prediction?')
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        except Exception as e:
            st.info('SHAP explanation unavailable for this prediction.')
            
            
            
st.divider()
st.caption(
    '⚠️ This tool is not a medical device and does not provide medical advice. '
    'Predictions are based on a limited dataset and should not be used for '
    'diagnosis or treatment decisions. Always consult a qualified healthcare '
    'professional if you have concerns about your health.'
)