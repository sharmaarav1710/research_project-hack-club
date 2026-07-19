import os
import io
import numpy as np
import pandas as pd
import gradio as gr
import xgboost as xgb
import shap
import matplotlib
# Use the non-interactive Agg backend to prevent GUI thread crashes on Vercel
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fastapi import FastAPI

# Initialize a FastAPI app instance for Vercel's serverless router
app = FastAPI()

def parse_hp_obo(filepath):
    hpo_dict = {}
    current_id = None
    current_name = None
    if not os.path.exists(filepath):
        return hpo_dict
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("[Term]"):
                current_id = None
                current_name = None
            elif line.startswith("id: HP:"):
                current_id = line.split("id: ")[1].split()[0]
            elif line.startswith("name: ") and current_id:
                current_name = line.split("name: ")[1]
                hpo_dict[current_id] = f"{current_id} ({current_name})"
    return hpo_dict

# Vercel will read files relative to the project root directory
HPO_OBO_PATH = os.path.join(os.getcwd(), "hp.obo")
HPO_HP_PATH = os.path.join(os.getcwd(), "phenotype.hpoa")

hpo_mapper = parse_hp_obo(HPO_OBO_PATH)

# Load data safely
if os.path.exists(HPO_HP_PATH):
    df_raw = pd.read_csv(HPO_HP_PATH, sep="\t", comment="#", low_memory=False)
    df_filtered = df_raw[df_raw["database_id"].str.startswith(('OMIM', 'ORPHA'))].copy()
    df_filtered['hpo_name'] = df_filtered['hpo_id'].map(hpo_mapper).fillna(df_filtered['hpo_id'])
    top_diseases = df_filtered['disease_name'].value_counts().index[:50]
    df_subset = df_filtered[df_filtered['disease_name'].isin(top_diseases)]
    matrix = df_subset.pivot_table(index='disease_name', columns='hpo_name', aggfunc='size', fill_value=0)
    matrix = (matrix > 0).astype(int)
    symptom_frequencies = matrix.sum(axis=0)
    matrix_filtered = matrix.loc[:, symptom_frequencies > 1]
    
    X_base = matrix_filtered.values
    disease_list = list(matrix_filtered.index)
    HPO_FEATURES = list(matrix_filtered.columns)
else:
    X_base = np.zeros((2, 2))
    disease_list = ["No Data Available", "Check File Paths"]
    HPO_FEATURES = ["Feature A", "Feature B"]

# Simulate Patient Cohorts
X_augmented = []
y_augmented = []
np.random.seed(42)
SAMPLES_PER_DISEASE = 30
for idx, base_vector in enumerate(X_base):
    symptom_indices = np.where(base_vector == 1)[0]
    for _ in range(SAMPLES_PER_DISEASE):
        patient_vector = base_vector.copy()
        if len(symptom_indices) > 1:
            drop_ratio = np.random.uniform(0.1, 0.3)
            num_to_drop = max(1, int(len(symptom_indices) * drop_ratio))
            dropped_indices = np.random.choice(symptom_indices, size=num_to_drop, replace=False)
            patient_vector[dropped_indices] = 0
        X_augmented.append(patient_vector)
        y_augmented.append(idx)

X = np.array(X_augmented)
y = np.array(y_augmented)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train classifiers globally so they compile during startup container initialization
lr_model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
lr_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=150, random_state=42)
rf_model.fit(X_train, y_train)

xgb_model = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train)

models = {'Logistic Regression': lr_model, 'Random Forest': rf_model, 'XGBoost': xgb_model}
metrics = {}

for name, model in models.items():
    preds = model.predict(X_test)
    metrics[name] = {
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds, average='macro', zero_division=0),
        'Recall': recall_score(y_test, preds, average='macro', zero_division=0),
        'F1-Score': f1_score(y_test, preds, average='macro', zero_division=0)
    }

def generate_shap_plot(model_name, input_vector):
    plt.clf()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    selected_model = models[model_name]

    if model_name == "Logistic Regression":
        coefs = selected_model.coef_[0]
        shap_vals = coefs * input_vector[0]
    else:
        explainer = shap.TreeExplainer(selected_model)
        shap_vals_all = explainer.shap_values(input_vector)
        if isinstance(shap_vals_all, list):
            shap_vals = shap_vals_all[0][0]
        elif len(shap_vals_all.shape) == 3:
            shap_vals = shap_vals_all[0, :, 0]
        else:
            shap_vals = shap_vals_all[0]
    
    indices = np.argsort(np.abs(shap_vals))[-10:]
    top_vals = shap_vals[indices]
    top_features = [HPO_FEATURES[i] for i in indices]
    
    colors = ['#ff0051' if val >= 0 else '#008bfb' for val in top_vals]
    ax.barh(np.arange(len(top_vals)), top_vals, align='center', color=colors)
    ax.set_yticks(np.arange(len(top_vals)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel('SHAP Feature Attribution (Risk Factor Impact)')
    ax.set_title(f'{model_name} Local Symptom Attribution Map')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close()
    return Image.open(buf)

def clinician_diagnose(selected_model, symptom_checklist):
    input_vector = np.zeros((1, len(HPO_FEATURES)))
    if symptom_checklist:
        for symptom in symptom_checklist:
            if symptom in HPO_FEATURES:
                idx = HPO_FEATURES.index(symptom)
                input_vector[0, idx] = 1

    model = models[selected_model]
    probabilities = model.predict_proba(input_vector)[0]

    results = {disease_list[i]: float(probabilities[i]) for i in range(len(disease_list))}
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True)[:5])

    shap_chart = generate_shap_plot(selected_model, input_vector)
    return sorted_results, shap_chart

# Build Gradio UI
with gr.Blocks(title="Rare Disease Diagnostics Portal") as demo:
    gr.Markdown("# 🧬 Rare Disease Clinical Diagnostic Support Interface")
    gr.Markdown("A computational framework processing physical HPO ontologies to map clinical presentations directly to candidate diagnoses.")
    
    with gr.Tab("Diagnostic Assistant"):
        with gr.Row():
            with gr.Column(scale=1):
                model_selector = gr.Radio(choices=list(models.keys()), label="Active AI Engine Model Selection", value="Random Forest")
                symptom_input = gr.Dropdown(choices=HPO_FEATURES, multiselect=True, label="Patient HPO Symptoms (Searchable)")
                submit_btn = gr.Button("Analyze Symptoms", variant='primary')
            with gr.Column(scale=2):
                prediction_output = gr.Label(label="Top Predicted Rare Disease Candidates", num_top_classes=5)
                shap_output = gr.Image(type="pil", label="Local Feature Attribution (SHAP Bar Plot)")
        
        submit_btn.click(fn=clinician_diagnose, inputs=[model_selector, symptom_input], outputs=[prediction_output, shap_output])
        
    with gr.Tab("Benchmarking Reports"): 
        gr.Markdown("### Real Validation Metrics")
        df_metrics = pd.DataFrame(metrics).T.reset_index().rename(columns={'index': 'Model'})
        gr.DataFrame(df_metrics) 

# Mount the Gradio interface onto the FastAPI application at the root route
app = gr.mount_gradio_app(app, demo, path="/")