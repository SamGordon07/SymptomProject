import pandas as pd
import numpy as np
import joblib
import os

def load_models(model_dir='models'):
    feature_cols = joblib.load(f'{model_dir}/feature_columns.pkl')
    trained_diseases = joblib.load(f'{model_dir}/trained_diseases.pkl')
    confidence_tiers = joblib.load(f'{model_dir}/confidence_tiers.pkl')

    disease_models = {}
    for disease in trained_diseases:
        safe_name = disease.replace(' ', '_').replace('/', '_').replace("'", '')
        path = f'{model_dir}/disease_models/{safe_name}.pkl'
        if os.path.exists(path):
            disease_models[disease] = joblib.load(path)

    return feature_cols, disease_models, confidence_tiers


def predict_diseases(user_features, disease_models, feature_cols, top_n=5):
    input_df = pd.DataFrame([user_features])[feature_cols]

    scores = {}
    for disease, model in disease_models.items():
        prob_positive = model.predict_proba(input_df)[0][1]
        scores[disease] = prob_positive

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(disease, round(prob * 100, 1)) for disease, prob in ranked[:top_n]]