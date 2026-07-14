import pandas as pd
import numpy as np
import json
import os

train_path = "data/train.csv"
if not os.path.exists(train_path):
    print("train.csv not found")
    exit(1)

df = pd.read_csv(train_path)

# Apply Phase 4 Cleaning (correcting type of MSSubClass and removing outliers)
df["MSSubClass"] = df["MSSubClass"].astype(str)
df = df.drop(df[(df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000)].index)

# Apply Phase 5 Feature Engineering
def engineer_features(df_eng):
    df_eng = df_eng.copy()
    df_eng["HouseAge"] = df_eng["YrSold"] - df_eng["YearBuilt"]
    df_eng["HouseAge"] = df_eng["HouseAge"].apply(lambda x: max(0, x))
    
    df_eng["RemodelAge"] = df_eng["YrSold"] - df_eng["YearRemodAdd"]
    df_eng["RemodelAge"] = df_eng["RemodelAge"].apply(lambda x: max(0, x))
    df_eng["IsRemodeled"] = (df_eng["YearRemodAdd"] != df_eng["YearBuilt"]).astype(int)
    
    df_eng["TotalBathrooms"] = (
        df_eng["FullBath"] + 
        0.5 * df_eng["HalfBath"] + 
        df_eng.get("BsmtFullBath", 0) + 
        0.5 * df_eng.get("BsmtHalfBath", 0)
    )
    
    df_eng["TotalPorchArea"] = (
        df_eng.get("OpenPorchSF", 0) + 
        df_eng.get("EnclosedPorch", 0) + 
        df_eng.get("3SsnPorch", 0) + 
        df_eng.get("ScreenPorch", 0) + 
        df_eng.get("WoodDeckSF", 0)
    )
    
    df_eng["TotalSF"] = df_eng["GrLivArea"] + df_eng.get("TotalBsmtSF", 0)
    df_eng["HasGarage"] = (df_eng.get("GarageArea", 0) > 0).astype(int)
    df_eng["HasPool"] = (df_eng.get("PoolArea", 0) > 0).astype(int)
    df_eng["QualityScore"] = df_eng["OverallQual"] * df_eng["OverallCond"]
    df_eng["LuxuryHouse"] = ((df_eng["OverallQual"] >= 8) & (df_eng["GrLivArea"] > 2500)).astype(int)
    
    return df_eng

df_eng = engineer_features(df)
X = df_eng.drop(columns=["Id", "SalePrice"], errors="ignore")

# Find medians for numeric and modes for categoricals
defaults = {}
categorical_choices = {}
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=[np.object_]).columns.tolist()

for col in X.columns:
    if col in numeric_cols:
        # Fill missing with median
        val = X[col].median()
        if pd.isna(val):
            val = 0
        defaults[col] = float(val)
    else:
        # Fill missing with mode
        val = X[col].mode().iloc[0] if len(X[col].mode()) > 0 else "None"
        defaults[col] = str(val)
        categorical_choices[col] = [str(x) for x in X[col].dropna().unique().tolist()]

metadata = {
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "defaults": defaults,
    "categorical_choices": categorical_choices
}

with open("notebooks/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Metadata extracted successfully! Numeric cols:", len(numeric_cols), "Categorical cols:", len(categorical_cols))
