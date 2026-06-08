import pandas as pd
import numpy as np
import sklearn.preprocessing
import joblib
import os

# Fungsi preprocessing lengkap untuk Customer Shopping Dataset (Unsupervised Learning)
def preprocessing_pipeline(csv_path):
    df = pd.read_csv(csv_path)

    # Membuat direktori untuk menyimpan model preprocessor jika belum ada
    os.makedirs("preprocessing/model", exist_ok=True)

    # 1. Drop fitur yang tidak digunakan
    # Customer ID dihilangkan karena hanya merupakan identitas unik dan tidak bermakna untuk klastering
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
            
            # Format nama file agar aman (menghilangkan spasi dan tanda kurung)
            safe_name = feature.replace(' ', '_').replace('(', '').replace(')', '')
            joblib.dump(scaler, f"preprocessing/model/scaler_{safe_name}.joblib")
        return data

    df = scaling(numerical_columns, df)

    # 3. Encoding fitur kategorikal
    # Mengambil semua kolom yang bertipe 'object' (teks)
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    def encoding(features, data):
        for feature in features:
            encoder = sklearn.preprocessing.LabelEncoder()
            encoder.fit(data[feature])
            data[feature] = encoder.transform(data[feature])
            
            # Format nama file agar aman
            safe_name = feature.replace(' ', '_')
            joblib.dump(encoder, f"preprocessing/model/encoder_{safe_name}.joblib")
        return data

    df = encoding(categorical_columns, df)

    # Pengembalian full dataset yang sudah siap masuk ke model K-Means
    return df

if __name__ == "__main__":
    file_dataset = r"C:\Users\LENOVO\Desktop\kulyeahhhhhh\Pijak\MSML_Firda-Azzahra\Eksperimen_SML_Firda-Azzahra\Customer_shopping_dataset\shopping_trends.csv"
    df_clean = preprocessing_pipeline(file_dataset)
    print("Bentuk data setelah preprocessing:", df_clean.shape)

    # SIMPAN KE CSV
    Customer_shopping_preprocessing = "df_clean"
    os.makedirs(Customer_shopping_preprocessing, exist_ok=True)
    
    # Simpan file ke dalam folder tersebut
    Customer_shopping_preprocessing = os.path.join(Customer_shopping_preprocessing, "shopping_trends_preprocessed.csv")
    df_clean.to_csv(r"C:\Users\LENOVO\Desktop\kulyeahhhhhh\Pijak\MSML_Firda-Azzahra\Eksperimen_SML_Firda-Azzahra\preprocessing\Customer_shopping_preprocessing\shopping_trends_preprocessed.csv", index=False)

    print(f"Bentuk data setelah preprocessing: {df_clean.shape}")
    print(f"File berhasil disimpan di: {Customer_shopping_preprocessing}")