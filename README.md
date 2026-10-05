# Eksperimen_SML_Ivan

Repository eksperimen data untuk submission akhir kelas Dicoding **"Membangun Sistem Machine Learning"**.

## Dataset

[Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail) — data transaksi online retail berbasis UK, periode Desember 2010 – Desember 2011 (541,909 baris transaksi).

## Use Case

**Prediksi Customer Churn** menggunakan pendekatan **RFM (Recency, Frequency, Monetary)**:
- Data transaksi diagregasi per customer menjadi fitur Recency, Frequency, dan Monetary
- Customer dilabeli *churn* jika tidak bertransaksi lagi dalam 90 hari terakhir dari tanggal referensi dataset

## Struktur Repository
Eksperimen_SML_Ivan/
├── .github/workflows/
│ └── preprocess.yml # CI untuk menjalankan preprocessing otomatis
├── online_retail_raw/
│ └── Online Retail.xlsx # dataset mentah
├── preprocessing/
│ ├── Eksperimen_Ivan.ipynb # notebook eksperimen manual (loading, EDA, cleaning, RFM)
│ ├── automate_Ivan.py # script preprocessing otomatis
│ ├── requirements.txt
│ └── online_retail_preprocessing/
│ ├── X_train.csv
│ ├── X_test.csv
│ ├── y_train.csv
│ └── y_test.csv
└── README.md

## Tahapan Preprocessing

1. **Data Cleaning** — buang baris dengan `Quantity`/`UnitPrice` tidak valid, `CustomerID` kosong, invoice cancelled (`C`), dan duplikat
2. **Feature Engineering RFM** — hitung Recency, Frequency, Monetary per customer
3. **Labeling** — churn jika Recency > 90 hari
4. **Transformasi** — log-transform (`log1p`) untuk Frequency & Monetary (right-skewed)
5. **Split & Scaling** — train-test split (80/20, stratified) + `StandardScaler`

## Menjalankan Preprocessing Otomatis

```bash
cd preprocessing
pip install -r requirements.txt
python automate_Ivan.py
```

Output akan tersimpan di `preprocessing/online_retail_preprocessing/`.

## CI Workflow

Workflow `.github/workflows/preprocess.yml` otomatis menjalankan `automate_Ivan.py` setiap ada perubahan pada dataset mentah atau script preprocessing, lalu commit ulang hasil dataset terbaru. Bisa juga dipicu manual lewat tab **Actions** di GitHub.