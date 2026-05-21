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