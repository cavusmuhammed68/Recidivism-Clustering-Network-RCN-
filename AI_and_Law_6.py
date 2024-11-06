# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 09:11:59 2024

@author: cavus
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE
from keras_tuner import RandomSearch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import shap
import os

# Make a directory to save figures
if not os.path.exists("figures"):
    os.makedirs("figures")

# Load the dataset
data = pd.read_csv('C:/Users/cavus/Desktop/AI and Law Paper/Megans_List.csv')

# Data Preprocessing
def calculate_age(dob):
    try:
        dob = datetime.strptime(dob, "%m-%d-%Y")
        today = datetime.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    except:
        return np.nan

data['Age'] = data['Date of Birth'].apply(calculate_age)

# Encode categorical columns using OneHotEncoder for better performance in deep learning models
onehot_encoder = OneHotEncoder(sparse=False)
columns_to_encode = ['Ethnicity', 'Sex', 'Eye Color', 'Hair Color']
encoded_columns = onehot_encoder.fit_transform(data[columns_to_encode])

# Handle height in feet and inches
def height_to_inches(height):
    try:
        match = re.match(r"(\d+)'(\d+)", height)
        if match:
            feet = int(match.group(1))
            inches = int(match.group(2))
            return feet * 12 + inches
        return np.nan
    except:
        return np.nan

data['Height'] = data['Height'].apply(height_to_inches)

# Handle missing values
numeric_columns = ['Age', 'Height', 'Weight']
imputer_num = SimpleImputer(strategy='mean')
data[numeric_columns] = imputer_num.fit_transform(data[numeric_columns])

# Create new features
def clean_year(year):
    try:
        return int(year)
    except:
        return np.nan

data['Year of Last Conviction'] = data['Year of Last Conviction'].apply(clean_year)
current_year = datetime.today().year
data['Time Since Last Conviction'] = current_year - data['Year of Last Conviction']
data['Multiple Offenses'] = data['Offense Code'].apply(lambda x: 1 if len(str(x).split(';')) > 1 else 0)

# Create recidivism label
data['Year of Last Release'] = data['Year of Last Release'].apply(clean_year)
data['Recidivism'] = data.apply(lambda row: 1 if row['Year of Last Release'] > row['Year of Last Conviction'] else 0, axis=1)

# Drop rows with missing target variable data
data = data.dropna(subset=['Recidivism'])

# Feature Scaling
scaler = StandardScaler()
numeric_features_scaled = scaler.fit_transform(data[numeric_columns])

# Combine encoded and scaled features
X = np.hstack((numeric_features_scaled, encoded_columns))
y = data['Recidivism']

# Handle Data Imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Deep Learning Model - Neural Network with Keras
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Dense(units=hp.Int('units', min_value=16, max_value=128, step=16),
                                 activation='relu', input_shape=(X_train.shape[1],)))
    model.add(keras.layers.Dense(units=hp.Int('units', min_value=16, max_value=128, step=16),
                                 activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3])),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Keras Tuner for Hyperparameter Tuning
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='tuner_results',
    project_name='recidivism_tuning'
)

# Run the hyperparameter search
tuner.search(X_train, y_train, epochs=10, validation_split=0.2)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the best model
best_model = tuner.hypermodel.build(best_hps)

# Train the best model
history = best_model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Model Evaluation
y_pred = (best_model.predict(X_test) > 0.5).astype("int32")

# Metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"ROC AUC Score: {roc_auc:.2f}")
print(f"F1 Score: {f1:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16})  # Adjust the font size here
plt.title('Confusion Matrix - Recidivism Prediction')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig("figures/confusion_matrix.png", dpi=600)
plt.show()


# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig("figures/roc_curve.png", dpi=600)
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred)
plt.plot(recall, precision, marker='.')
plt.title('Precision-Recall Curve')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig("figures/precision_recall_curve.png", dpi=600)
plt.show()

# Learning Curves (Accuracy and Loss)
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Learning Curves - Accuracy')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.savefig("figures/learning_curve_accuracy.png", dpi=600)
plt.show()

# Loss Curve
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Learning Curves - Loss')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.savefig("figures/learning_curve_loss.png", dpi=600)
plt.show()

# SHAP Explainability
explainer = shap.DeepExplainer(best_model, X_train)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
plt.savefig("figures/shap_summary_plot.png", dpi=600)

# Clustering Offenders using KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

# PCA for Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data['Cluster'], cmap='viridis', s=50)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Offender Clusters - PCA Visualization')
plt.colorbar(label='Cluster')
plt.savefig("figures/pca_clusters.png", dpi=600)
plt.show()

# t-SNE for Enhanced Clustering Visualization with Different Perplexities and Iterations
# Try multiple t-SNE configurations to compare clustering results

perplexities = [5, 30, 50]
iterations = [300, 500, 1000]

for perplexity in perplexities:
    for n_iter in iterations:
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
        X_tsne = tsne.fit_transform(X)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=data['Cluster'], cmap='plasma', s=50)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title(f't-SNE Visualization (Perplexity={perplexity}, Iterations={n_iter})')
        plt.colorbar(label='Cluster')
        
        # Save the figure with specified DPI
        filename = f"figures/tsne_perplexity_{perplexity}_iterations_{n_iter}.png"
        plt.savefig(filename, dpi=600)
        plt.show()

# Pairwise Scatter Plot of Key Features
# This visualization will help us understand how key features relate to each other across clusters

key_features = ['Age', 'Height', 'Weight', 'Time Since Last Conviction']

sns.pairplot(data[key_features + ['Cluster']], hue='Cluster', palette='viridis')
plt.suptitle('Pairwise Scatter Plot of Key Features Across Clusters', y=1.02)
plt.savefig("figures/pairwise_scatter_clusters.png", dpi=600)
plt.show()

# Box Plots to Visualize Feature Distributions Across Recidivism Classes
# Use box plots to show the distribution of Age, Height, and Time Since Last Conviction
# across the Recidivism classes (0 or 1)

plt.figure(figsize=(12, 8))
sns.boxplot(x='Recidivism', y='Age', data=data)
plt.title('Age Distribution Across Recidivism Classes')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig("figures/boxplot_age_recidivism.png", dpi=600)
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(x='Recidivism', y='Height', data=data)
plt.title('Height Distribution Across Recidivism Classes')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig("figures/boxplot_height_recidivism.png", dpi=600)
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(x='Recidivism', y='Time Since Last Conviction', data=data)
plt.title('Time Since Last Conviction Distribution Across Recidivism Classes')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig("figures/boxplot_time_conviction_recidivism.png", dpi=600)
plt.show()

# SHAP Summary Plot for Feature Importance (Already included earlier but now with save option)
shap.summary_plot(shap_values, X_test)
plt.savefig("figures/shap_summary_plot.png", dpi=600)






import matplotlib.pyplot as plt
import seaborn as sns

# Confusion Matrix
plt.figure(figsize=(12, 8))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
#plt.title('Confusion Matrix - Recidivism Prediction', fontsize=15)
plt.xlabel('Predicted', fontsize=15)
plt.ylabel('Actual', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig("figures/confusion_matrix.png", dpi=600)
plt.show()

# ROC Curve
plt.figure(figsize=(12, 8))
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--')
#plt.title('ROC Curve', fontsize=15)
plt.xlabel('False Positive Rate', fontsize=15)
plt.ylabel('True Positive Rate', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc="lower right", fontsize=15)
plt.savefig("figures/roc_curve.png", dpi=600)
plt.show()

# Precision-Recall Curve
plt.figure(figsize=(12, 8))
precision, recall, _ = precision_recall_curve(y_test, y_pred)
plt.plot(recall, precision, marker='.')
#plt.title('Precision-Recall Curve', fontsize=16)
plt.xlabel('Recall', fontsize=16)
plt.ylabel('Precision', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig("figures/precision_recall_curve.png", dpi=600)
plt.show()

# Learning Curves (Accuracy)
plt.figure()
plt.figure(figsize=(12, 8))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
#plt.title('Learning Curves - Accuracy', fontsize=16)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='best', fontsize=16)
plt.savefig("figures/learning_curve_accuracy.png", dpi=600)
plt.show()

# Learning Curves (Loss)
plt.figure()
plt.figure(figsize=(12, 8))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
#plt.title('Learning Curves - Loss', fontsize=16)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='best', fontsize=16)
plt.savefig("figures/learning_curve_loss.png", dpi=600)
plt.show()

# PCA Clustering Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data['Cluster'], cmap='viridis', s=50)
#plt.title('Offender Clusters - PCA Visualization', fontsize=16)
plt.xlabel('PCA Component 1', fontsize=16)
plt.ylabel('PCA Component 2', fontsize=16)
plt.colorbar(label='Cluster')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig("figures/pca_clusters.png", dpi=600)
plt.show()

# Pairwise Scatter Plot of Key Features
sns.pairplot(data[key_features + ['Cluster']], hue='Cluster', palette='viridis')
plt.suptitle('Pairwise Scatter Plot of Key Features Across Clusters', y=1.02, fontsize=16)
plt.savefig("figures/pairwise_scatter_clusters.png", dpi=600)
plt.show()

# Boxplots to Visualize Feature Distributions Across Recidivism Classes

# Age
plt.figure(figsize=(12, 8))
sns.boxplot(x='Recidivism', y='Age', data=data)
#plt.title('Age Distribution Across Recidivism Classes', fontsize=16)
plt.xlabel('Recidivism', fontsize=16)
plt.ylabel('Age', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig("figures/boxplot_age_recidivism.png", dpi=600)
plt.show()

# Height
plt.figure(figsize=(12, 8))
sns.boxplot(x='Recidivism', y='Height', data=data)
#plt.title('Height Distribution Across Recidivism Classes', fontsize=16)
plt.xlabel('Recidivism', fontsize=16)
plt.ylabel('Height', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig("figures/boxplot_height_recidivism.png", dpi=600)
plt.show()

# Time Since Last Conviction
plt.figure(figsize=(12, 8))
sns.boxplot(x='Recidivism', y='Time Since Last Conviction', data=data)
#plt.title('Time Since Last Conviction Distribution Across Recidivism Classes', fontsize=16)
plt.xlabel('Recidivism', fontsize=16)
plt.ylabel('Time Since Last Conviction', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig("figures/boxplot_time_conviction_recidivism.png", dpi=600)
plt.show()

# t-SNE for Enhanced Clustering Visualization with Different Perplexities and Iterations
perplexities = [5, 30, 50]
iterations = [300, 500, 1000]

for perplexity in perplexities:
    for n_iter in iterations:
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
        X_tsne = tsne.fit_transform(X)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=data['Cluster'], cmap='plasma', s=50)
        plt.title(f't-SNE Visualization (Perplexity={perplexity}, Iterations={n_iter})', fontsize=16)
        plt.xlabel('t-SNE Component 1', fontsize=16)
        plt.ylabel('t-SNE Component 2', fontsize=16)
        plt.colorbar(label='Cluster')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        
        filename = f"figures/tsne_perplexity_{perplexity}_iterations_{n_iter}.png"
        plt.savefig(filename, dpi=600)
        plt.show()



























