Rare Disease Center

Its an  clinical decision support platform designed to analyze rare disease phenotypes, predict genetic symptom patterns, and provide explainable AI insights.

Project Overview
The proposed research will use machine learning algorithms to discover the hidden vectors of rare diseases based on clinical symptoms. In order to achieve this, the system will use the patients’ phenotypes to compute possible diagnoses, risks, and then employ explainable artificial intelligence such as SHAP to display important symptoms for each diagnosis.

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

AI Usage Declaration

Ai has been used to help in the generation of boilerplate code for the UI layout, creation of basic data, and debugging. All the algorithmic architecture and machine learning models (Gradient Boosting Classifier among others) have been selected and implemented independently for clinical significance.

Application Interface

Here is a visual overview of the interface in action:

Initial dashboard view before symptom input:
![Initial Interface View](sc1%20project%201.png)

Active diagnosis prediction panel demonstrating feature attribution mapping:
![Active Diagnostic Analysis](sc2%20project%202.png)





