"""
Training script for K-Means clustering (5 clusters) and Linear Regression
to predict Premium Amount. Saves trained models into model/ directory.

Usage:
    python train_model.py --csv_path /path/to/customer_segmentation_data.csv
"""
# Set Matplotlib backend sebelum mengimpor pyplot untuk menghindari isu GUI di server
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import argparse
import os
import joblib
from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def load_and_prepare_data(csv_path: str):
    """
    Loads data, performs initial cleaning, feature engineering (Recency),
    and drops irrelevant columns.
    """
    df = pd.read_csv(csv_path)

    # Visualisasi boxplot untuk eksplorasi data numerik
    for col in ['Age', 'Income Level', 'Coverage Amount', 'Premium Amount']:
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, y=col, color='skyblue')
        plt.title(f'Boxplot of {col}')
        plt.tight_layout()
        plt.close() # Penting untuk menutup plot

    # --- Feature Engineering: Recency ---
    if "Purchase History" in df.columns:
        df["Purchase History"] = pd.to_datetime(df["Purchase History"], errors='coerce')
        analysis_date = df["Purchase History"].max()
        df["Recency"] = (analysis_date - df["Purchase History"]).dt.days
        # PERBAIKAN: Mengisi nilai NaN pada Recency menggunakan metode yang direkomendasikan Pandas
        df["Recency"] = df["Recency"].fillna(0) # Mengganti inplace=True dengan assignment
    else:
        df["Recency"] = 0

    # Menghapus kolom yang tidak relevan untuk training model
    if "Customer ID" in df.columns:
        df = df.drop(columns=["Customer ID"])
    
    # Menghapus kolom 'Purchase History' setelah 'Recency' dihitung
    if "Purchase History" in df.columns:
        df = df.drop(columns=["Purchase History"])

    # Menghapus baris yang mengandung NaN setelah semua pra-pemrosesan awal
    df.dropna(inplace=True)

    return df

def build_preprocess_pipeline(df):
    """
    Builds a ColumnTransformer for preprocessing features.
    For this simplified version, only numerical features for regression are considered.
    """
    numeric_cols_for_regression = ["Age", "Income Level", "Coverage Amount", "Recency"]
    categorical_cols_for_regression = [] 

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols_for_regression),
        ],
        remainder='passthrough' 
    )
    return preprocessor, numeric_cols_for_regression

def train_models(df, preprocessor):
    """
    Trains K-Means clustering model and Linear Regression model.
    """
    # ----------------- CLUSTERING (Unsupervised Learning) -----------------
    cluster_features = ["Age", "Income Level", "Coverage Amount", "Recency"]
    X_clustering = df[cluster_features]

    if X_clustering.isnull().sum().sum() > 0:
        raise ValueError("NaNs found in X_clustering features after preprocessing. Please check data loading/cleaning.")

    scaler = StandardScaler()
    X_scaled_for_kmeans = scaler.fit_transform(X_clustering)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(X_scaled_for_kmeans)

    # ----------------- REGRESSION (Supervised Learning) -----------------
    reg_features = ["Age", "Income Level", "Coverage Amount", "Recency"]
    X_reg = df[reg_features] 
    y_reg = df["Premium Amount"] 

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    reg_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()), 
        ('regressor', LinearRegression())
    ])

    reg_pipeline.fit(X_train_reg, y_train_reg)

    return (kmeans, scaler), reg_pipeline

def main():
    """Main function to parse arguments, load data, train models, and save them."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True, help="Path to customer_segmentation_data.csv")
    parser.add_argument("--out_dir", default="model", help="Directory to save .pkl files")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_and_prepare_data(args.csv_path)
    
    preprocessor, _ = build_preprocess_pipeline(df) 
    
    kmeans_model, reg_model = train_models(df, preprocessor)

    joblib.dump(kmeans_model[0], os.path.join(args.out_dir, "kmeans_model.pkl"))
    joblib.dump(kmeans_model[1], os.path.join(args.out_dir, "scaler.pkl"))
    joblib.dump(reg_model, os.path.join(args.out_dir, "reg_model.pkl"))
    print("✅ Model berhasil disimpan ke folder", args.out_dir)

if __name__ == "__main__":
    main()
