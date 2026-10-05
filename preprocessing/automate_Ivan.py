import pandas as pd
import numpy as np
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_online_retail(raw_path, churn_threshold=90, test_size=0.2, random_state=42):
    """
    Preprocessing otomatis dataset Online Retail -> RFM features + label churn.

    Params:
        raw_path (str): path ke file Online Retail.xlsx
        churn_threshold (int): batas hari recency untuk label churn
        test_size (float): proporsi data test
        random_state (int): seed untuk reproducibility

    Returns:
        X_train, X_test, y_train, y_test (pd.DataFrame / pd.Series)
    """
    # 1. Load data
    df = pd.read_excel(raw_path)

    # 2. Cleaning
    df_clean = df[
        (df['Quantity'] > 0) &
        (df['UnitPrice'] > 0) &
        (df['CustomerID'].notnull())
    ].copy()
    df_clean = df_clean.drop_duplicates()
    df_clean = df_clean[~df_clean['InvoiceNo'].astype(str).str.startswith('C')]
    df_clean['CustomerID'] = df_clean['CustomerID'].astype(int)
    df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']

    # 3. RFM feature engineering
    reference_date = df_clean['InvoiceDate'].max() + dt.timedelta(days=1)
    rfm = df_clean.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('TotalPrice', 'sum')
    ).reset_index()

    # 4. Label churn
    rfm['Churn'] = (rfm['Recency'] > churn_threshold).astype(int)

    # 5. Log transform
    rfm['Recency_log'] = np.log1p(rfm['Recency'])
    rfm['Frequency_log'] = np.log1p(rfm['Frequency'])
    rfm['Monetary_log'] = np.log1p(rfm['Monetary'])

    # 6. Split
    X = rfm[['Recency_log', 'Frequency_log', 'Monetary_log']]
    y = rfm['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 7. Scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_online_retail('../online_retail_raw/Online Retail.xlsx')

    import os
    os.makedirs('online_retail_preprocessing', exist_ok=True)
    X_train.to_csv('online_retail_preprocessing/X_train.csv', index=False)
    X_test.to_csv('online_retail_preprocessing/X_test.csv', index=False)
    y_train.to_csv('online_retail_preprocessing/y_train.csv', index=False)
    y_test.to_csv('online_retail_preprocessing/y_test.csv', index=False)
    print("Preprocessing selesai. File tersimpan di online_retail_preprocessing/")
