from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import os
import joblib 
import matplotlib # Perbaikan: Set backend sebelum import pyplot
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 
import seaborn as sns 
from io import BytesIO 
import base64 
from datetime import datetime 

app = Flask(__name__)
app.secret_key = "supersecretkey" 

KMEANS_MODEL_PATH = os.path.join("model", "kmeans_model.pkl")
REG_MODEL_PATH = os.path.join("model", "reg_model.pkl")
SCALER_PATH = os.path.join("model", "scaler.pkl")

kmeans = None
scaler = None
reg_model = None
try:
    kmeans = joblib.load(KMEANS_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    reg_model = joblib.load(REG_MODEL_PATH)
    print("✅ Model berhasil dimuat.")
except FileNotFoundError:
    print("⚠️ File model tidak ditemukan. Harap latih model terlebih dulu menggunakan train_model.py.")
except Exception as e:
    print(f"⚠️ Kesalahan saat memuat file model: {e}. Harap latih model terlebih dulu menggunakan train_model.py.")

NUMERIC_FEATURES = ["Age", "Income Level", "Coverage Amount", "Recency"]

EXPECTED_REGRESSION_COLUMNS = ["Age", "Income Level", "Coverage Amount", "Recency"]


def format_currency(value):
    """Memformat nilai numerik ke format Rupiah Indonesia."""
    return f"Rp {value:,.0f}".replace(",", ".")

def preprocess_input(form_data):
    """
    Membuat DataFrame dari input formulir, hanya dengan fitur-fitur yang disederhanakan.
    """
    input_data = {}
    column_mapping = {
        "age": "Age",
        "income_level": "Income Level",
        "coverage_amount": "Coverage Amount",
        "purchase_history": "Purchase History" 
    }

    for form_field, col_name in column_mapping.items():
        value = form_data.get(form_field)
        if col_name in ["Age", "Income Level", "Coverage Amount"]:
            try:
                input_data[col_name] = [float(value)]
            except (ValueError, TypeError):
                input_data[col_name] = [0.0] 
        elif col_name == "Purchase History":
            input_data[col_name] = [value if value else "2023-01-01"] 
    
    df_input = pd.DataFrame(input_data)

    if "Purchase History" in df_input.columns:
        df_input["Purchase History"] = pd.to_datetime(df_input["Purchase History"], errors='coerce')
        analysis_date = pd.Timestamp.now()
        df_input["Recency"] = (analysis_date - df_input["Purchase History"]).dt.days
        df_input["Recency"].fillna(0, inplace=True) 
    else:
        df_input["Recency"] = 0
        
    if "Purchase History" in df_input.columns:
        df_input = df_input.drop(columns=["Purchase History"])
    
    final_cols_for_regressor = NUMERIC_FEATURES 
    
    for col in final_cols_for_regressor:
        if col not in df_input.columns:
            df_input[col] = 0.0

    df_input = df_input[final_cols_for_regressor]

    return df_input

# --- Fungsi utilitas untuk menghasilkan visualisasi ---
def generate_plot_image(plot_func, **kwargs):
    plt.figure(figsize=(10, 6))
    plot_func(**kwargs)
    plt.tight_layout()
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close()
    plot_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    return plot_base64

def load_data_for_visualizations():
    """Memuat dan pra-proses data untuk tujuan visualisasi."""
    try:
        df = pd.read_csv("customer_segmentation_data.csv")
        df.dropna(inplace=True) 

        if "Purchase History" in df.columns:
            df["Purchase History"] = pd.to_datetime(df["Purchase History"], errors='coerce')
            analysis_date = df["Purchase History"].max() 
            df["Recency"] = (analysis_date - df["Purchase History"]).dt.days
            df["Recency"].fillna(0, inplace=True)
        else:
            df["Recency"] = 0

        if "Customer ID" in df.columns:
            df.drop(columns=["Customer ID"], inplace=True)
        
        columns_to_drop_for_viz_data = ["Segmentation Group", "Purchase History"]
        df = df.drop(columns=[col for col in columns_to_drop_for_viz_data if col in df.columns], errors='ignore')

        return df
    except Exception as e:
        print(f"Kesalahan saat memuat data untuk visualisasi: {e}")
        flash("Gagal memuat data untuk visualisasi.", "error")
        return pd.DataFrame()

# --- Rute Aplikasi Flask ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    cluster_descriptions = {
        0: "Pelanggan berusia muda dengan penghasilan tinggi, cenderung memilih premi dan cakupan perlindungan yang besar.",
        1: "Pelanggan usia menengah dengan penghasilan sedang, memilih premi terjangkau dan perlindungan moderat.",
        2: "Pelanggan berusia lebih tua dengan penghasilan rendah, cenderung memilih premi kecil dan perlindungan minimal.",
        3: "Pelanggan muda dengan penghasilan terbatas namun memilih premi yang tinggi, kemungkinan memiliki preferensi perlindungan lebih.",
        4: "Pelanggan usia menengah ke atas dengan penghasilan tinggi, memilih premi dan cakupan yang seimbang dan realistis."
    }

    if kmeans is None or reg_model is None or scaler is None:
        flash("Model belum dilatih atau file model tidak ditemukan. Harap jalankan train_model.py terlebih dulu.", "error")
        return redirect(url_for("index"))

    full_input_df = preprocess_input(request.form)

    # --- Prediksi K-Means ---
    kmeans_input_data = full_input_df[NUMERIC_FEATURES]
    if kmeans_input_data.isnull().sum().sum() > 0:
        flash("Data input numerik untuk segmentasi mengandung nilai yang hilang. Harap periksa input.", "error")
        print("NaNs found in kmeans_input_data:", kmeans_input_data.isnull().sum())
        return redirect(url_for("index"))

    kmeans_data_scaled = scaler.transform(kmeans_input_data)
    cluster_label = int(kmeans.predict(kmeans_data_scaled)[0])
    cluster_desc = cluster_descriptions.get(cluster_label, "Deskripsi tidak tersedia untuk cluster ini.")

    # --- Prediksi Regresi ---
    premi_pred = 0.0
    try:
        premi_pred = float(reg_model.predict(full_input_df)[0]) 
    except Exception as e:
        flash(f"Terjadi kesalahan saat memprediksi premi: {e}", "error")
        print(f"Kesalahan selama prediksi regresi: {e}")
        print("DataFrame untuk regresi (head):", full_input_df.head())
        print("Kolom yang diharapkan oleh regressor:", NUMERIC_FEATURES) 
        print("Kolom aktual di full_input_df:", full_input_df.columns.tolist())
        return redirect(url_for("index"))

    return render_template("result.html",
                           cluster=cluster_label,
                           cluster_desc=cluster_desc,
                           premi=format_currency(round(premi_pred, 0)))

@app.route('/visualisasi/boxplot')
def show_boxplot():
    """Menampilkan boxplot untuk kolom-kolom numerik."""
    df_viz = load_data_for_visualizations()
    if df_viz.empty:
        return redirect(url_for('index'))

    plot_urls = {}
    numeric_cols_for_boxplot = ['Age', 'Income Level', 'Coverage Amount', 'Premium Amount']

    for col in numeric_cols_for_boxplot:
        if col in df_viz.columns and not df_viz[col].isnull().all():
            plot_base64 = generate_plot_image(
                lambda data_df_param, column_name_param: sns.boxplot(data=data_df_param, y=column_name_param, color='skyblue'),
                data_df_param=df_viz, column_name_param=col
            )
            plot_urls[col.replace(" ", "_").lower() + '_boxplot'] = plot_base64
        else:
            print(f"Kolom '{col}' tidak ditemukan atau semua nilainya NaN untuk boxplot.")

    return render_template('visualisasi.html', plot_title='Boxplot Data Numerik', plots=plot_urls, plot_type='multi_boxplot')


@app.route('/visualisasi/cluster')
def show_cluster_plot():
    """Menampilkan scatterplot hasil clustering."""
    if kmeans is None or scaler is None:
        flash("Model K-Means atau scaler belum dimuat. Harap latih model terlebih dulu.", "error")
        return redirect(url_for('index'))

    df_viz = load_data_for_visualizations()
    if df_viz.empty:
        return redirect(url_for('index'))

    if all(col in df_viz.columns for col in NUMERIC_FEATURES) and not df_viz[NUMERIC_FEATURES].isnull().values.any():
        X_viz_scaled = scaler.transform(df_viz[NUMERIC_FEATURES])
        df_viz['Cluster'] = kmeans.predict(X_viz_scaled)
    else:
        flash("Kolom yang diperlukan untuk clustering mengandung nilai yang hilang atau tidak lengkap dalam data visualisasi.", "error")
        print("Missing values in cluster features:", df_viz[NUMERIC_FEATURES].isnull().sum())
        return redirect(url_for('index'))

    # PERBAIKAN: Ambil sampel 25 baris secara acak untuk visualisasi scatterplot
    # Pastikan jumlah baris lebih dari atau sama dengan 25 sebelum sampling
    if len(df_viz) >= 25:
        df_viz_sampled = df_viz.sample(n=25, random_state=42) # random_state untuk hasil yang dapat direproduksi
    else:
        df_viz_sampled = df_viz # Jika data kurang dari 25, gunakan semua yang ada

    plot_base64 = generate_plot_image(
        lambda data_df_param: sns.scatterplot(data=data_df_param, x='Income Level', y='Premium Amount', hue='Cluster', palette='tab10'),
        data_df_param=df_viz_sampled # Gunakan DataFrame yang sudah disampel
    )
    return render_template('visualisasi.html', plot_title='Visualisasi Cluster (Pendapatan vs Premi)', plot_image=plot_base64, plot_type='single_plot')

@app.route('/visualisasi/radar_chart')
def show_radar_chart():
    if kmeans is None or scaler is None:
        flash("Model K-Means atau scaler belum dimuat. Harap latih model terlebih dulu.", "error")
        return redirect(url_for('index'))

    df_viz = load_data_for_visualizations()
    if df_viz.empty:
        return redirect(url_for('index'))

    if all(col in df_viz.columns for col in NUMERIC_FEATURES) and not df_viz[NUMERIC_FEATURES].isnull().values.any():
        X_viz_scaled = scaler.transform(df_viz[NUMERIC_FEATURES])
        df_viz['Cluster'] = kmeans.predict(X_viz_scaled)
    else:
        flash("Kolom yang diperlukan untuk clustering mengandung nilai yang hilang atau tidak lengkap dalam data visualisasi.", "error")
        return redirect(url_for('index'))

    cluster_profiles = df_viz.groupby('Cluster')[NUMERIC_FEATURES].mean()

    def create_radar_chart_plot_func(profiles_param, features_list_param):
        labels = features_list_param
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for idx, row in profiles_param.iterrows():
            values = row.tolist()
            values += values[:1]
            ax.plot(angles, values, label=f'Cluster {idx}', linewidth=1.5)
            ax.fill(angles, values, alpha=0.15)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.set_title("Radar Chart Profil Cluster", va='bottom', fontsize=16)
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    
    plot_base64 = generate_plot_image(
        create_radar_chart_plot_func,
        profiles_param=cluster_profiles, features_list_param=NUMERIC_FEATURES
    )
    return render_template('visualisasi.html', plot_title='Radar Chart Profil Cluster', plot_image=plot_base64, plot_type='single_plot')

@app.route('/visualisasi/actual_predicted')
def show_actual_predicted_plot():
    if reg_model is None:
        flash("Model regresi belum dimuat. Harap latih model terlebih dulu.", "error")
        return redirect(url_for('index'))

    df_viz = load_data_for_visualizations()
    if df_viz.empty:
        return redirect(url_for('index'))

    X_reg_viz = pd.DataFrame()
    y_reg_viz = pd.Series()

    columns_to_drop_for_viz_reg = ["Premium Amount", "Segmentation Group"]
    
    if "Premium Amount" in df_viz.columns:
        X_reg_viz = df_viz.drop(columns=[col for col in columns_to_drop_for_viz_reg if col in df_viz.columns])
        y_reg_viz = df_viz["Premium Amount"]
    else:
        flash("Kolom 'Premium Amount' tidak ditemukan dalam data visualisasi.", "error")
        return redirect(url_for('index'))
    
    y_pred_viz = []
    try:
        X_reg_viz_simplified = X_reg_viz[NUMERIC_FEATURES]
        y_pred_viz = reg_model.predict(X_reg_viz_simplified)
    except Exception as e:
        flash(f"Terjadi kesalahan saat memprediksi untuk visualisasi aktual vs prediksi: {e}. Pastikan kolom input sesuai.", "error")
        print(f"Kesalahan selama prediksi untuk plot aktual vs prediksi: {e}")
        print("DataFrame X_reg_viz_simplified untuk regresi (head):", X_reg_viz_simplified.head())
        print("Kolom yang diharapkan oleh regressor:", NUMERIC_FEATURES)
        print("Kolom aktual di X_reg_viz_simplified:", X_reg_viz_simplified.columns.tolist())
        return redirect(url_for('index'))

    plot_base64 = generate_plot_image(
        lambda y_actual_param, y_predicted_param, y_min_param, y_max_param: (
            sns.scatterplot(x=y_actual_param, y=y_predicted_param, alpha=0.6, color="deeppink"),
            plt.plot([y_min_param, y_max_param], [y_min_param, y_max_param], 'k--', lw=2),
            plt.xlabel("Actual Premium"),
            plt.ylabel("Predicted Premium"),
            plt.title("Actual vs Predicted Premium Amount")
        ),
        y_actual_param=y_reg_viz, y_predicted_param=y_pred_viz,
        y_min_param=y_reg_viz.min(), y_max_param=y_reg_viz.max()
    )
    return render_template('visualisasi.html', plot_title='Actual vs Predicted Premium Amount', plot_image=plot_base64, plot_type='single_plot')

if __name__ == "__main__":
    app.run(debug=True)
