Rare Disease Clinical Command Center

Its an interactive clinical decision support platform designed to analyze rare disease phenotypes, predict genetic symptom patterns, and provide explainable AI insights.

Project Overview

This research project leverages advanced machine learning models to identify underlying rare disease vectors from clinical symptom profiles. By utilizing patient phenotype data, the system evaluates potential diagnoses, calculates risk scores, and uses explainable AI techniques like SHAP to show clinicians which symptoms most strongly influence each prediction.

Key Features

Automated Phenotype Matching: Maps input symptoms directly to human phenotype ontology terms.
Predictive Disease Modeling: Employs tree-based machine learning classifiers to assess high-risk rare disease matches.
Explainable AI Diagnostic Support: Generates visual feature importance charts to explain the mathematical model's reasoning.
Interactive Clinical Interface: A web-accessible dashboard built with Gradio for seamless medical professional interaction.

Dataset and Architecture

The platform uses curated clinical symptom ontologies alongside simulated and anonymized patient records. 
Phenotypic classification is powered by a high-performance Gradient Boosting Classifier.
Feature impact is analyzed and rendered using SHAP (SHapley Additive exPlanations).
Real-time web deployment is managed via a containerized Python service on Render.

Project Structure

app.py: The main entry point containing the data pipelines, model configurations, and the user interface code.
requirements.txt: Defines all required library dependencies including scikit-learn, xgboost, shap, and gradio.
hp.obo / phenotype.hpoa: Standardized Human Phenotype Ontology database files used for semantic matching.

Deployment Details

Live URL: https://research-project-hack-club.onrender.com
Runtime: Python 3
Hosting Platform: Render Free Tier Web Service

Local Setup Instructions

1. Clone this repository to your local machine.
2. Install all required dependencies by running: pip install -r requirements.txt
3. Start the application locally with: python app.py
4. Open the local link generated in your terminal to interact with the interface.

Thank you!!!