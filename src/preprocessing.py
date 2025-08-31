import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_data(df):
    df = df.copy()
    df['J'] = LabelEncoder().fit_transform(df['J'])
    df['Monto'] = df['Monto'].replace(',', '', regex=True).astype(float)

    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].str.replace(',', '').astype(float)
            except:
                pass
    df = df.fillna(-999)
    return df
