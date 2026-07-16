import pytest
import pandas as pd
from train import engineer_features

def test_engineer_features_calculation():
    # Construct a sample raw input DataFrame
    raw_data = {
        "YearBuilt": [2000, 2005],
        "YearRemodAdd": [2000, 2010],
        "YrSold": [2008, 2008],
        "FullBath": [2, 1],
        "HalfBath": [1, 0],
        "BsmtFullBath": [1, 0],
        "BsmtHalfBath": [0, 1],
        "OpenPorchSF": [10, 0],
        "EnclosedPorch": [20, 5],
        "3SsnPorch": [0, 0],
        "ScreenPorch": [0, 0],
        "WoodDeckSF": [50, 0],
        "GrLivArea": [1500, 2600],
        "TotalBsmtSF": [1000, 1200],
        "GarageArea": [500, 0],
        "PoolArea": [0, 100],
        "OverallQual": [6, 9],
        "OverallCond": [5, 6]
    }
    df_raw = pd.DataFrame(raw_data)
    
    # Run feature engineering
    df_eng = engineer_features(df_raw)
    
    # Assertions
    # 1. House Age & Remodel Age
    # House 1: Age = 2008 - 2000 = 8. Remodel age = 8. IsRemodeled = 0
    assert df_eng.loc[0, "HouseAge"] == 8
    assert df_eng.loc[0, "RemodelAge"] == 8
    assert df_eng.loc[0, "IsRemodeled"] == 0
    
    # House 2: Age = 2008 - 2005 = 3. Remodel age = 2008 - 2010 = -2 (max(0, -2) = 0). IsRemodeled = 1
    assert df_eng.loc[1, "HouseAge"] == 3
    assert df_eng.loc[1, "RemodelAge"] == 0
    assert df_eng.loc[1, "IsRemodeled"] == 1
    
    # 2. Total Bathrooms
    # House 1: FullBath(2) + 0.5*HalfBath(1) + BsmtFullBath(1) + 0.5*BsmtHalfBath(0) = 3.5
    assert df_eng.loc[0, "TotalBathrooms"] == 3.5
    # House 2: 1 + 0 + 0 + 0.5 = 1.5
    assert df_eng.loc[1, "TotalBathrooms"] == 1.5
    
    # 3. Total Porch Area
    # House 1: Open(10) + Enclosed(20) + 3Ssn(0) + Screen(0) + WoodDeck(50) = 80
    assert df_eng.loc[0, "TotalPorchArea"] == 80
    
    # 4. Total SF
    assert df_eng.loc[0, "TotalSF"] == 2500
    
    # 5. HasGarage & HasPool
    assert df_eng.loc[0, "HasGarage"] == 1
    assert df_eng.loc[0, "HasPool"] == 0
    assert df_eng.loc[1, "HasGarage"] == 0
    assert df_eng.loc[1, "HasPool"] == 1
    
    # 6. Quality Score
    assert df_eng.loc[0, "QualityScore"] == 30
    
    # 7. Luxury House Indicator
    # House 1: Qual=6 < 8 -> Luxury=0
    assert df_eng.loc[0, "LuxuryHouse"] == 0
    # House 2: Qual=9 >= 8 and GrLivArea=2600 > 2500 -> Luxury=1
    assert df_eng.loc[1, "LuxuryHouse"] == 1
