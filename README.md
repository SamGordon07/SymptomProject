# Symptom Checker

A machine learning web app that takes symptoms + patient profile data as input and outputs the most likely conditions ranked by confidence.

## Live App
[Link will go here after deployment]

## Project Overview
Built using a One vs Rest Random Forest approach — training a separate binary classifier per disease — allowing the model to learn both what a disease looks like when confirmed and what it looks like when ruled out. The full data pipeline runs from a SQLite database through feature engineering and model training to a live deployed Streamlit app, with every user query logged back to the database. Built as a portfolio project to demonstrate the complete data science workflow from raw data to deployed product.

Due to prior experience working with data in healthcare, this project was a natural transition into professional data science workflow.

## Tech Stack
- Python + Jupyter Labs
- SQLite + sqlite3 (data storage, formatting, and query logging)
- pandas, numpy (data processing and analysis)
- scikit-learn (Random Forest — One vs Rest approach)
- SHAP (model explainability)
- Streamlit (web app)

## Approach
A One vs Rest strategy was chosen over standard multiclass classification because of how the dataset is structured. With 116 diseases and only 349 rows, many diseases appear only a handful of times. A single multiclass model (like Native Random Forest, which was initially planned to be used) struggles in this situation because it tries to learn all diseases simultaneously and gets dominated by the more common ones. Training a separate binary Random Forest per disease means each model focuses entirely on one condition — learning its specific combination of features — rather than competing against 115 other classes at once.

The dataset includes an Outcome Variable column indicating whether a diagnosis was confirmed (Positive) or ruled out (Negative) for each patient record. Rather than dropping this column or using it as a prediction target, it was used to construct smarter training labels. For each disease model, rows where that disease was present and confirmed Positive were labelled 1, while all other rows — including cases where that disease was present but returned Negative — were labelled 0. This means each model learns not just what symptoms point toward a disease but also what a ruled-out case of that disease looks like, giving it more honest and complete signal than training on confirmed cases alone. Using binary classification fit seamlessly with this process.

The SHAP summary plot shows how each input feature influenced the model's predictions across the entire training population, not just for a single user. Each dot represents one patient record, colored by whether their feature value was high or low, and positioned on the x-axis by how much it pushed the prediction toward or away from that disease. The confidence tier system exists because not all 77 trained disease models are equally reliable — a model trained on 20 confirmed cases is far more trustworthy than one trained on 2. Each prediction is tagged as High, Medium, Low, or Very Low quality based on how many positive training examples backed that disease model, so users can see at a glance how much weight to give each result.

## Model Performance
- 77 disease models trained (out of 116 total)
- 39 diseases skipped due to zero positive training examples
- Average AUC (chance that the model correctly places a positive/correct disgnosis over a negative/incorrect one): 0.622 across evaluated diseases (those with at least 3 positive cases — this ensures that 3-fold cross validation can be used to maintain accuracy while still keeping a variety of diseases in the dataset)

## Dataset
Disease Symptom and Patient Profile Dataset from Kaggle. 349 patient records, 116 diseases, 4 symptoms plus patient profile data (age, gender, blood pressure, cholesterol) and outcome variable.

Link to Kaggle dataset here --> https://www.kaggle.com/datasets/uom190346a/disease-symptoms-and-patient-profile-dataset?resource=download

## Known Limitations
- Only 4 binary symptoms limits predictive power
- 39 diseases have no confirmed positive examples in the dataset
- AUC of 0.622 reflects the ceiling of what is learnable from this feature set rather than a modeling failure
- Not validated on real clinical data
- For educational purposes only

## How to Run Locally
- git clone https://github.com/SamGordon07/symptom-checker
- cd symptom-checker
- python -m venv venv
- source venv/bin/activate  
    - Windows: venv\Scripts\activate
- pip install -r requirements.txt
- streamlit run app/main.py

## Project Structure
    symptom-checker/
    ├── data/
    ├── notebooks/
    ├── src/
    ├── app/
    ├── models/
    └── README.md
