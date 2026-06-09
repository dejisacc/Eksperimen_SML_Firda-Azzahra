import pandas as pd
import numpy as np
import sklearn.preprocessing
import joblib
import os

# Fungsi preprocessing lengkap untuk Customer Shopping Dataset (Unsupervised Learning)
def preprocessing_pipeline(csv_path, base_dir):
    df = pd.read_csv(csv_path)

    # buat folder 'model' 
    model_dir = os.path.join(base_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    # 1. Drop fitur yang tidak digunakan
    if 'Customer ID' in df.columns:
        df = df.drop(columns=['Customer ID'], axis=1)

    # 2. Scaling data numerikal
    numerical_columns = [
        'Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases'
    ]

    def scaling(features, data):
        for feature in features:
            scaler = sklearn.preprocessing.StandardScaler()
            scaler.fit(data[[feature]])
            data[feature] = scaler.transform(data[[feature]])
            
            # Format nama file agar aman
            safe_name = feature.replace(' ', '_').replace('(', '').replace(')', '')
            
            # Simpan tepat ke dalam folder model
            joblib.dump(scaler, os.path.join(model_dir, f"scaler_{safe_name}.joblib"))
        return data

    df = scaling(numerical_columns, df)

    # 3. Encoding fitur kategorikal
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    def encoding(features, data):
        for feature in features:
            encoder = sklearn.preprocessing.LabelEncoder()
            encoder.fit(data[feature])
            data[feature] = encoder.transform(data[feature])
            
            # Format nama file agar aman
            safe_name = feature.replace(' ', '_')
            
            # Simpan tepat ke dalam folder model
            joblib.dump(encoder, os.path.join(model_dir, f"encoder_{safe_name}.joblib"))
        return data

    df = encoding(categorical_columns, df)

    return df

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    file_dataset = os.path.abspath(os.path.join(BASE_DIR, "..", "Customer_shopping_dataset", "shopping_trends.csv"))
    
    if not os.path.exists(file_dataset):
        file_dataset = os.path.abspath(os.path.join(BASE_DIR, "..", "shopping_trends.csv"))

    if not os.path.exists(file_dataset):
        print(f"Peringatan: Dataset tidak ditemukan di rute:\n{file_dataset}")
    else:
        print("Memulai proses preprocessing data...")
        df_clean = preprocessing_pipeline(file_dataset, BASE_DIR)

        # SIMPAN DATA BERSIH KE CSV 
        lokasi_simpan = os.path.join(BASE_DIR, "shopping_trends_preprocessed.csv")
        df_clean.to_csv(lokasi_simpan, index=False)
        
        print(f"\n[SUKSES] Bentuk data setelah preprocessing: {df_clean.shape}")
        print(f"Semua file berhasil disimpan dengan rapi di dalam folder:\n{BASE_DIR}")
