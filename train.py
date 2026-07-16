import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

def engineer_features(df):
    df_eng = df.copy()
    
    # Calculate house age and remodeling age
    df_eng["HouseAge"] = df_eng["YrSold"] - df_eng["YearBuilt"]
    df_eng["HouseAge"] = df_eng["HouseAge"].apply(lambda x: max(0, x))
    
    df_eng["RemodelAge"] = df_eng["YrSold"] - df_eng["YearRemodAdd"]
    df_eng["RemodelAge"] = df_eng["RemodelAge"].apply(lambda x: max(0, x))
    df_eng["IsRemodeled"] = (df_eng["YearRemodAdd"] != df_eng["YearBuilt"]).astype(int)
    
    # Calculate total bathroom count
    df_eng["TotalBathrooms"] = (
        df_eng["FullBath"] + 
        0.5 * df_eng["HalfBath"] + 
        df_eng.get("BsmtFullBath", 0) + 
        0.5 * df_eng.get("BsmtHalfBath", 0)
    )
    
    # Calculate total porch area
    df_eng["TotalPorchArea"] = (
        df_eng.get("OpenPorchSF", 0) + 
        df_eng.get("EnclosedPorch", 0) + 
        df_eng.get("3SsnPorch", 0) + 
        df_eng.get("ScreenPorch", 0) + 
        df_eng.get("WoodDeckSF", 0)
    )
    
    # Calculate additional interaction and boolean features
    df_eng["TotalSF"] = df_eng["GrLivArea"] + df_eng.get("TotalBsmtSF", 0)
    df_eng["HasGarage"] = (df_eng.get("GarageArea", 0) > 0).astype(int)
    df_eng["HasPool"] = (df_eng.get("PoolArea", 0) > 0).astype(int)
    df_eng["QualityScore"] = df_eng["OverallQual"] * df_eng["OverallCond"]
    df_eng["LuxuryHouse"] = ((df_eng["OverallQual"] >= 8) & (df_eng["GrLivArea"] > 2500)).astype(int)
    
    return df_eng

def train_model(data_path="data/train.csv", model_output_path="notebooks/house_price_pipeline.joblib", metadata_output_path="notebooks/model_metadata.json"):
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found at {data_path}")
        
    df = pd.read_csv(data_path)
    
    # Phase 4 Cleaning (casting MSSubClass to string & removing outliers)
    df["MSSubClass"] = df["MSSubClass"].astype(str)
    outliers = df[(df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000)]
    print(f"Removing {len(outliers)} outliers (GrLivArea > 4000 & SalePrice < 300000)...")
    df = df.drop(outliers.index)
    
    # Phase 5 Feature Engineering
    print("Engineering features...")
    df_eng = engineer_features(df)
    
    # Split features and target
    X = df_eng.drop(columns=["Id", "SalePrice"], errors="ignore")
    y = np.log1p(df_eng["SalePrice"])
    
    # Determine columns by pre-processing type
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=[np.object_]).columns.tolist()
    
    # Extract defaults and metadata
    print("Generating model metadata...")
    defaults = {}
    categorical_choices = {}
    
    for col in X.columns:
        if col in numeric_cols:
            val = X[col].median()
            defaults[col] = float(0 if pd.isna(val) else val)
        else:
            val = X[col].mode().iloc[0] if len(X[col].mode()) > 0 else "None"
            defaults[col] = str(val)
            categorical_choices[col] = [str(x) for x in X[col].dropna().unique().tolist()]
            
    metadata = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "defaults": defaults,
        "categorical_choices": categorical_choices
    }
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(metadata_output_path), exist_ok=True)
    
    # Define Pipelines
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, numeric_cols),
        ("cat", cat_pipeline, categorical_cols)
    ])
    
    # Final Model Pipeline
    print("Training XGBoost pipeline...")
    regressor = XGBRegressor(
        objective="reg:squarederror",
        enable_categorical=True,
        learning_rate=0.1,
        max_depth=3,
        n_estimators=150,
        n_jobs=-1,
        random_state=42
    )
    
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])
    
    pipeline.fit(X, y)
    
    # Serialize model and metadata
    print(f"Saving model to {model_output_path}...")
    joblib.dump(pipeline, model_output_path)
    
    print(f"Saving metadata to {metadata_output_path}...")
    with open(metadata_output_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print("Retraining completed successfully!")

if __name__ == "__main__":
    train_model()
