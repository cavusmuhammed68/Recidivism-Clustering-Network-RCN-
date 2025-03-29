# Recidivism Clustering Network (RCN)

A transparent and bias-resilient AI framework for predicting recidivism using deep learning, clustering (k-means, PCA, t-SNE), SMOTE, and explainable AI (SHAP).  
This repository supports the following academic study:

**"Transparent and Bias-Resilient AI Framework for Recidivism Prediction Using Deep Learning and Clustering Techniques in Criminal Justice"**

---

## 📚 About the Project

The RCN framework improves recidivism prediction by combining:
- Deep learning for classification
- SMOTE for addressing class imbalance
- Clustering techniques for subgroup identification
- SHAP (SHapley Additive exPlanations) for explainable, legally compliant predictions

The goal is to build a fair, transparent, and interpretable model for use in criminal justice settings.

---

## 📁 Repository Structure


---

## ⚙️ Installation

1. Clone the repository:
bash
git clone https://github.com/yourusername/Recidivism-Clustering-Network-RCN.git
cd Recidivism-Clustering-Network-RCN

Install required packages:
pip install -r requirements.txt

🧹 Data Preprocessing
Dataset used: Publicly available data from California Megan’s Law registry.
Open notebooks/1_preprocessing.ipynb
This notebook performs:
Outlier handling (e.g., age > 100 years)
Missing value imputation
One-hot encoding of categorical features
Scaling of numerical features
SMOTE to balance the dataset
Output files are saved in the data/processed/ directory.

🏋️ Model Training
Open notebooks/2_training.ipynb
This notebook trains a deep neural network using the processed dataset
Hyperparameter tuning is included using Keras Tuner
The trained model is saved in the models/ folder
Metrics reported:
Accuracy: ~75%
Precision, Recall, F1-Score

📈 Explainability & Clustering
Open notebooks/3_explainability.ipynb

SHAP is used to generate:
Feature importance plots
Force plots for individual cases
Clustering (PCA + t-SNE + k-means) is used to:
Identify hidden subgroups
Visualize offender patterns
Detect bias in subgroup predictions

✅ Key Features
Combines machine learning with legal fairness principles
Addresses class imbalance with SMOTE
Provides explainable predictions using SHAP
Visualizes offender groups using PCA and t-SNE
Designed for real-world deployment in legal workflows

📜 Citation
If you use this repository in your research, please cite:

Cavus, M., et al. (2025).
Transparent and Bias-Resilient AI Framework for Recidivism Prediction Using Deep Learning and Clustering Techniques in Criminal Justice.
Submitted to Applied Soft Computing.


