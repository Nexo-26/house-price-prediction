# Valuation AI — House Price Prediction Engine

An advanced, end-to-end Machine Learning pipeline and web application designed to forecast residential property values based on historical Ames, Iowa housing data using a serialized **XGBoost** regression model.

---

## Directory Structure

```text
house_price_prediction/
├── data/
│   ├── download_data.py      # Script to download dataset CSVs
│   ├── train.csv             # Training dataset
│   └── test.csv              # Test dataset
├── frontend/
│   ├── static/
│   │   ├── css/style.css     # Glassmorphic visual style sheets
│   │   └── js/main.js        # Form handler, UI animations & API integration
│   ├── templates/
│   │   └── index.html        # Main web dashboard interface
│   └── app.py                # Flask server with prediction endpoints & logging
├── notebooks/
│   ├── house_price_prediction.ipynb   # Complete 16-phase Jupyter Notebook
│   ├── house_price_pipeline.joblib    # Serialized scikit-learn model pipeline
│   ├── model_metadata.json            # Extracted feature schemas & default values
│   ├── generate_notebook.py           # Programmatic notebook generator script
│   └── images/                        # Visual graphs generated in EDA & Interpretability
├── tests/
│   ├── test_features.py      # Unit tests for engineered features
│   └── test_app.py           # Integration tests for Flask routing & endpoints
├── Dockerfile                # Production multi-stage Docker build config
├── .dockerignore             # Specifies files excluded from Docker container context
├── requirements.txt          # Explicit list of Python package dependencies
├── train.py                  # Standalone CLI model retraining pipeline
└── README.md                 # Main setup & execution documentation
```

---

## Installation & Setup

Ensure you have **Python 3.8+** installed. Follow the steps below to setup:

1. **Clone/Navigate to the directory**:
   ```bash
   cd house_price_prediction
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment**:
   * **Windows (PowerShell)**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   * **Linux/macOS**:
     ```bash
     source venv/bin/activate
     ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Execution Guide

### 1. Download Dataset (Optional)
If raw data files are missing or need updating, run:
```bash
python data/download_data.py
```
This fetches the latest Ames housing datasets to `data/train.csv` and `data/test.csv`.

### 2. Retrain the Machine Learning Model
To perform data cleaning, compute feature engineering, fit the pipeline, and export model outputs (`house_price_pipeline.joblib` and `model_metadata.json`):
```bash
python train.py
```

### 3. Run Automated Tests
Execute the automated test suite using `pytest`:
```bash
pytest tests/
```

### 4. Run the Web Application Local Server
Launch the Flask development server:
```bash
python frontend/app.py
```
Open a browser and navigate to `http://127.0.0.1:5000` to interact with the Valuation AI dashboard.

---

## Docker Containerization

To package and run the application in a lightweight container:

1. **Build the Docker Image**:
   ```bash
   docker build -t house-price-app .
   ```

2. **Run the Container**:
   ```bash
   docker run -p 5000:5000 house-price-app
   ```
   Access the web app at `http://localhost:5000`.
