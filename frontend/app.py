import os
import json
import joblib
import logging
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Configure standard logging to console
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)

app = Flask(__name__)

# Locate paths relative to the current script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, '..', 'notebooks', 'house_price_pipeline.joblib')
METADATA_PATH = os.path.join(SCRIPT_DIR, '..', 'notebooks', 'model_metadata.json')

# Load the saved model pipeline and metadata
model = None
metadata = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    logging.info("Successfully loaded model pipeline.")
else:
    logging.warning(f"Model not found at {MODEL_PATH}")

if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    logging.info("Successfully loaded model metadata.")
else:
    logging.warning(f"Metadata not found at {METADATA_PATH}")

# Hardcoded neighborhood average sale prices computed from training dataset
NEIGHBORHOOD_AVERAGES = {
    'Blmngtn': 194870.88, 'Blueste': 137500.00, 'BrDale': 104493.75, 
    'BrkSide': 124834.05, 'ClearCr': 212565.43, 'CollgCr': 197965.77, 
    'Crawfor': 210624.73, 'Edwards': 128219.70, 'Gilbert': 192854.51, 
    'IDOTRR': 100123.78, 'MeadowV': 98576.47, 'Mitchel': 156270.12, 
    'NAmes': 145847.08, 'NPkVill': 142694.44, 'NWAmes': 189050.07, 
    'NoRidge': 335295.32, 'NridgHt': 316270.62, 'OldTown': 128225.30, 
    'SWISU': 142591.36, 'Sawyer': 136793.14, 'SawyerW': 186555.80, 
    'Somerst': 225379.84, 'StoneBr': 310499.00, 'Timber': 242247.45, 
    'Veenker': 238772.73
}

def engineer_features(df_eng):
    df_eng = df_eng.copy()
    
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

@app.route('/')
def home():
    # Render static homepage
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not model or not metadata:
            logging.error("Model or metadata not loaded on server.")
            return jsonify({'error': 'Model or metadata not loaded on server.'}), 500
        
        user_input = request.json
        if not user_input:
            logging.warning("Prediction endpoint called with empty payload.")
            return jsonify({'error': 'No input data provided.'}), 400
        
        logging.info(f"Prediction requested for neighborhood: {user_input.get('Neighborhood', 'unknown')}")
        
        # 1. Start with the baseline default record from metadata
        full_record = metadata['defaults'].copy()
        
        # 2. Update with the user's specific entries
        for key, val in user_input.items():
            if key in full_record:
                # Cast numeric inputs correctly
                if isinstance(full_record[key], float):
                    full_record[key] = float(val)
                elif isinstance(full_record[key], int):
                    full_record[key] = int(val)
                else:
                    full_record[key] = str(val)
        
        # Ensure YrSold is aligned if the house is newer than 2008
        if 'YearBuilt' in user_input:
            y_built = int(user_input['YearBuilt'])
            y_remod = int(user_input.get('YearRemodAdd', y_built))
            full_record['YrSold'] = max(2008, y_built, y_remod)

        # 3. Create DataFrame (required format for the scikit-learn pipeline)
        df_input = pd.DataFrame([full_record])
        
        # 4. Perform feature engineering
        df_engineered = engineer_features(df_input)
        
        # 5. Ensure the DataFrame has all required columns in the exact order
        # matching what ColumnTransformer was fit on
        all_feature_cols = metadata['numeric_cols'] + metadata['categorical_cols']
        df_final = df_engineered[all_feature_cols]
        
        # 6. Predict log price and back-transform to actual USD
        pred_log = model.predict(df_final)
        pred_usd = np.expm1(pred_log)[0]
        
        # 7. Add neighborhood comparison analysis
        neighborhood = user_input.get('Neighborhood', 'CollgCr')
        nh_avg = NEIGHBORHOOD_AVERAGES.get(neighborhood, 180000.0)
        percent_diff = ((pred_usd - nh_avg) / nh_avg) * 100
        
        response_data = {
            'predicted_price': float(round(pred_usd, 2)),
            'neighborhood': neighborhood,
            'neighborhood_avg': float(round(nh_avg, 2)),
            'percent_diff': float(round(percent_diff, 1))
        }
        
        logging.info(f"Prediction successful: ${response_data['predicted_price']} USD (diff vs average: {response_data['percent_diff']}%)")
        return jsonify(response_data)
        
    except Exception as e:
        logging.error("Error occurred during prediction processing:", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run server on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
