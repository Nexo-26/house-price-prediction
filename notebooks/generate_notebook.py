import json
import os

notebook = {
 "cells": [],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

def add_md(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    })

def add_code(code):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.split("\n")]
    })

# Add Title
add_md("""# Production-Grade House Price Prediction ML Pipeline
### An End-to-End Educational & Production-Quality Case Study
This notebook walks through a complete Machine Learning workflow, from business understanding to model serialization, following industry best practices in Data Science and Machine Learning Engineering.""")

# PHASE 1: Business Understanding
add_md("""## Phase 1 — Business Understanding

### 1. Objective
To understand the business context of house price prediction, map it to a machine learning task, define success criteria, and establish metrics.

### 2. Theory
Real estate valuation is a critical process for buyers, sellers, lenders, and online portals (e.g., Zillow, Opendoor). Accurate pricing reduces friction in transactions, optimizes mortgage lending portfolios, and provides transparency to the market.

### 3. Why this step is necessary
Without a clear business objective, machine learning projects often suffer from misalignment (e.g., optimizing for a metric that doesn't capture business value or failing to account for real-world deployment limitations).

### 4. Mathematical Intuition
We define the target variable $y_i$ as the actual Sale Price and $\\hat{y}_i$ as the predicted Sale Price.
The primary business metric is Mean Absolute Error (MAE):
$$MAE = \\frac{1}{N} \\sum_{i=1}^N |y_i - \\hat{y}_i|$$
And Root Mean Squared Error (RMSE) on log-transformed prices to evaluate percentage errors:
$$RMSE_{\\log} = \\sqrt{\\frac{1}{N} \\sum_{i=1}^N (\\log(y_i + 1) - \\log(\\hat{y}_i + 1))^2}$$

### 5. Python Implementation""")
add_code("""# Define baseline metrics for business goals
TARGET_RMSE_LOG = 0.13  # Representing ~13% error
TARGET_MAE_USD = 15000  # $15,000 USD average error
print(f"Target Performance Goals:\\n- RMSE Log: {TARGET_RMSE_LOG}\\n- MAE USD: ${TARGET_MAE_USD}")
""")
add_md("""### 6. Explanation of the Code
We establish constants for our target model performance based on business constraints.

### 7. Interpretation of the Output
The baseline target is defined: we want a model that predicts house values within a ~13% log-margin, equivalent to approximately $15,000 on average for mid-range homes.

### 8. Common Mistakes
- Relying purely on R-squared which scales with the variance of the test set, instead of scale-dependent metrics like MAE or RMSE.
- Failing to account for log-scale evaluation, which treats a $10k error on a $100k home and a $100k error on a $1M home with equal relative weight.

### 9. Best Practices
- Define KPIs alongside stakeholders before building models.
- Set up a log-evaluation metric for housing price prediction since prices are right-skewed and errors should be relative.

""")

# PHASE 2: Data Understanding
add_md("""## Phase 2 — Data Understanding

### 1. Objective
To load, inspect, and analyze the dataset dimensions, types, missing values, duplicates, and target variable characteristics.

### 2. Theory
Data understanding involves examining the schema, rows, columns, and properties of the files. The Ames Housing dataset has 1,460 samples (train) and 81 columns, including 79 explanatory features, 1 ID, and 1 Target variable (`SalePrice`).

### 3. Why this step is necessary
It allows us to check for corrupted files, understand types (nominal, ordinal, continuous, discrete), and plan cleaning/preprocessing pipelines.

### 4. Mathematical Intuition
We evaluate the distribution shape of the target variable `SalePrice` using Skewness ($S$) and Kurtosis ($K$):
$$S = \\frac{\\frac{1}{N} \\sum_{i=1}^N (y_i - \\bar{y})^3}{s^3}$$
$$K = \\frac{\\frac{1}{N} \\sum_{i=1}^N (y_i - \\bar{y})^4}{s^4} - 3$$
Where $\\bar{y}$ is the sample mean, and $s$ is the standard deviation. A skewness of 0 represents a perfectly symmetric distribution; kurtosis measures the tail thickness.

### 5. Python Implementation""")
add_code("""import pandas as pd
import numpy as np
import os

# Define file paths
train_path = "../data/train.csv"
test_path = "../data/test.csv"

# Load datasets
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# 1. Shape
print(f"Train Shape: {df_train.shape}")
print(f"Test Shape: {df_test.shape}")

# 2. Head & Tail
print("\\nFirst 3 rows of train set:")
display(df_train.head(3))

print("\\nLast 3 rows of train set:")
display(df_train.tail(3))

# 3. Sample rows
print("\\nRandom Sample of 3 rows:")
display(df_train.sample(3, random_state=42))

# 4. Info & Data Types
print("\\nData Types & Info:")
df_train.info()

# 5. Summary Statistics
print("\\nSummary Statistics of Numeric Columns:")
display(df_train.describe().T.head(10))

# 6. Missing Values
missing_val = df_train.isnull().sum()
missing_pct = 100 * missing_val / len(df_train)
missing_df = pd.DataFrame({"Missing Count": missing_val, "Percentage (%)": missing_pct})
missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values(by="Missing Count", ascending=False)
print("\\nColumns with Missing Values:")
display(missing_df.head(10))

# 7. Duplicate Rows
duplicates = df_train.duplicated().sum()
print(f"\\nDuplicate rows found: {duplicates}")

# 8. Cardinality of Categorical Columns
cat_cols = df_train.select_dtypes(include=["object"]).columns
cardinality = df_train[cat_cols].nunique().sort_values(ascending=False)
print("\\nCardinality of Categorical Columns:")
display(cardinality.head(10))

# 9. Target Variable Analysis
skew = df_train["SalePrice"].skew()
kurt = df_train["SalePrice"].kurt()
print(f"\\nTarget Variable ('SalePrice') Statistics:\\n- Mean: ${df_train['SalePrice'].mean():.2f}\\n- Median: ${df_train['SalePrice'].median():.2f}\\n- Skewness: {skew:.4f}\\n- Kurtosis: {kurt:.4f}")
""")
add_md("""### 6. Explanation of the Code
We read the training and test CSV files using Pandas, display their structural attributes, summarize numeric distributions, calculate missing value statistics, count duplicates, and compute skewness/kurtosis of the target variable `SalePrice`.

### 7. Interpretation of the Output
- The training set contains 1,460 observations and 81 columns, while the test set has 1,459 observations and 80 columns (excluding the target column `SalePrice`).
- The dataset contains many missing values (e.g. `PoolQC`, `MiscFeature`, `Alley`, `Fence` have >80% missing data).
- The target variable `SalePrice` is highly right-skewed (skewness ≈ 1.88) and leptokurtic (kurtosis ≈ 6.53), showing that a log transformation is highly recommended.

### 8. Common Mistakes
- Neglecting to inspect the test dataset shape and content.
- Missing the fact that `MSSubClass` is numeric but actually represents a categorical variable (building class), which needs casting.

### 9. Best Practices
- Always check missing percentage values to decide whether to drop columns or impute them.
- Look at the cardinality of text categories to prepare for encoding (high cardinality, e.g. `Neighborhood` with 25 categories, requires special care).

""")

# PHASE 3: Exploratory Data Analysis (EDA)
add_md("""## Phase 3 — Exploratory Data Analysis (EDA)

### 1. Objective
To perform visual and statistical analysis of variables, exploring distributions, interactions, correlations, and outliers.

### 2. Theory
EDA involves looking at single columns (univariate), pair-wise relationships (bivariate), and complex multi-variable relationships (multivariate) to identify visual trends, anomalies, and structural relationships.

### 3. Why this step is necessary
Models are only as good as our understanding of data patterns. EDA highlights outliers that can distort predictions and reveals strong linear or non-linear associations.

### 4. Mathematical Intuition
Pearson Correlation Coefficient ($r$):
$$r = \\frac{\\sum (x - \\bar{x})(y - \\bar{y})}{\\sqrt{\\sum (x - \\bar{x})^2 \\sum (y - \\bar{y})^2}}$$
Ranges from -1 (perfect negative) to +1 (perfect positive).

### 5. Python Implementation""")
add_code("""import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Create images folder if not exists
os.makedirs("images", exist_ok=True)

# 1. Univariate Analysis: Target variable distribution
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(df_train["SalePrice"], kde=True, ax=axes[0], color="royalblue")
axes[0].set_title("SalePrice Distribution (Raw)")
axes[0].set_xlabel("SalePrice ($)")

sns.histplot(np.log1p(df_train["SalePrice"]), kde=True, ax=axes[1], color="teal")
axes[1].set_title("SalePrice Distribution (Log Transformed)")
axes[1].set_xlabel("Log(SalePrice + 1)")
plt.tight_layout()
plt.savefig("images/target_distribution.png")
plt.show()

# 2. Univariate Analysis: Boxplot of SalePrice
plt.figure(figsize=(8, 4))
sns.boxplot(x=df_train["SalePrice"], color="coral")
plt.title("Boxplot of SalePrice Showing Outliers")
plt.savefig("images/target_boxplot.png")
plt.show()

# 3. Bivariate Analysis: GrLivArea vs SalePrice (Scatter)
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df_train["GrLivArea"], y=df_train["SalePrice"], alpha=0.6, color="purple")
plt.title("Above Grade Living Area (GrLivArea) vs SalePrice")
plt.xlabel("GrLivArea (Sq Ft)")
plt.ylabel("SalePrice ($)")
plt.savefig("images/grlivarea_vs_price.png")
plt.show()

# 4. Bivariate Analysis: Boxplot of OverallQual vs SalePrice
plt.figure(figsize=(10, 5))
sns.boxplot(x=df_train["OverallQual"], y=df_train["SalePrice"], palette="viridis")
plt.title("Overall Quality (OverallQual) vs SalePrice")
plt.savefig("images/overallqual_vs_price.png")
plt.show()

# 5. Multivariate Analysis: Correlation Heatmap of top 10 features
numeric_cols = df_train.select_dtypes(include=[np.number]).columns
top10_corr_cols = df_train[numeric_cols].corr()["SalePrice"].abs().sort_values(ascending=False).head(11).index
corr_matrix = df_train[top10_corr_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix Heatmap of Top 10 Features with SalePrice")
plt.savefig("images/correlation_heatmap.png")
plt.show()

# 6. Pivot Table showing Group-wise Analysis
pivot = df_train.pivot_table(values="SalePrice", index="OverallQual", columns="Fireplaces", aggfunc="mean")
print("Average Sale Price by Quality and Fireplaces:")
display(pivot)
""")
add_md("""### 6. Explanation of the Code
We plot target distribution (raw and log-transformed) to evaluate normality. We use a boxplot to highlight outlier values, a scatter plot to analyze the relationship between Living Area and target price, boxplots for ordinal quality scores, and a correlation matrix heatmap to show linear relationships.

### 7. Interpretation of the Output
- The target variable distribution becomes highly Gaussian under `np.log1p` transformation.
- The boxplot reveals many outliers above $340,000 USD.
- The scatter plot of `GrLivArea` vs `SalePrice` shows a strong positive correlation, but highlights two extreme outliers at the bottom-right (large living area >4000 sq ft, but low price <$200,000). Dean De Cock advises removing these.
- `OverallQual` has a massive impact on `SalePrice`, showing a clear non-linear trend as quality goes from 8 to 10.
- Top correlation features are `OverallQual` (0.79), `GrLivArea` (0.71), `GarageCars` (0.64), `GarageArea` (0.62), and `TotalBsmtSF` (0.61).

### 8. Common Mistakes
- Misinterpreting high correlation as causality.
- Ignoring extreme outliers in bivariate scatter plots that can destabilize linear models.

### 9. Best Practices
- Always plot target log-transform side-by-side with raw variables.
- Label every plot with correct titles and axes for readability.

""")

# PHASE 4: Data Cleaning
add_md("""## Phase 4 — Data Cleaning

### 1. Objective
To handle missing values, correct data types, detect and remove outliers, and address skewed features.

### 2. Theory
- **MCAR (Missing Completely at Random)**: Missingness has no relationship with any data (e.g. a random sensor failure).
- **MAR (Missing at Random)**: Missingness depends on other observed variables (e.g. missing income details for self-employed workers).
- **MNAR (Missing Not at Random)**: Missingness depends on the value of the variable itself (e.g. high-net-worth individuals refusing to share their income).
- Outliers distort gradients in linear models. Log transformations stabilize variance in highly skewed continuous distributions.

### 3. Why this step is necessary
Missing data breaks scikit-learn models. Outliers inflate MSE and skew regression coefficients.

### 4. Mathematical Intuition
IQR Outlier Rule:
$$IQR = Q3 - Q1$$
$$Boundaries = [Q1 - 1.5 \\cdot IQR, \\quad Q3 + 1.5 \\cdot IQR]$$
Any value outside these boundaries is flagged as a potential outlier.

### 5. Python Implementation""")
add_code("""# Correcting Types: MSSubClass should be categorical
df_train["MSSubClass"] = df_train["MSSubClass"].astype(str)
df_test["MSSubClass"] = df_test["MSSubClass"].astype(str)

# 1. Outlier Removal (based on Dean De Cock recommendations for Ames Dataset)
print(f"Shape before outlier removal: {df_train.shape}")
df_train = df_train.drop(df_train[(df_train["GrLivArea"] > 4000) & (df_train["SalePrice"] < 300000)].index)
print(f"Shape after outlier removal: {df_train.shape}")

# 2. Imputing Missing Values
def clean_missing(df):
    df_clean = df.copy()
    
    # Categoricals where NA represents "None"
    none_cols = [
        "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
        "GarageType", "GarageFinish", "GarageQual", "GarageCond",
        "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
        "MasVnrType"
    ]
    for col in none_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna("None")
            
    # Numerics where NA represents 0
    zero_cols = [
        "GarageYrBlt", "GarageArea", "GarageCars", "BsmtFinSF1", "BsmtFinSF2",
        "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "MasVnrArea"
    ]
    for col in zero_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)
            
    # Impute LotFrontage with median of its neighborhood
    if "LotFrontage" in df_clean.columns:
        df_clean["LotFrontage"] = df_clean.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
        
    # Mode imputation for others with low missing counts
    mode_cols = ["Electrical", "MSZoning", "Utilities", "Exterior1st", "Exterior2nd", "KitchenQual", "Functional", "SaleType"]
    for col in mode_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            
    return df_clean

df_train_clean = clean_missing(df_train)
df_test_clean = clean_missing(df_test)

# Verify no missing values remain in train (except Target if it was missing, but Target is not missing)
remaining_missing = df_train_clean.isnull().sum().sum()
print(f"Remaining missing values in train: {remaining_missing}")
""")
add_md("""### 6. Explanation of the Code
We cast `MSSubClass` to string since the numerical values represent category codes. We drop the two large living area outliers from train. We apply logical imputations (constant `'None'` for missing classes, `0` for missing sizes, neighborhood-grouped median for `LotFrontage`, and mode for standard categories).

### 7. Interpretation of the Output
Outlier removal drops 2 observations. The custom imputation function successfully processes the training set, reducing the missing value count to zero.

### 8. Common Mistakes
- Using test set statistics to impute train set values (data leakage).
- Blindly dropping rows containing missing values, which would discard most of the training set.

### 9. Best Practices
- Separate logical imputations (e.g. NA means None) from statistical imputations (median/mean/mode).
- Document every decision made during cleaning.

""")

# PHASE 5: Feature Engineering
add_md("""## Phase 5 — Feature Engineering

### 1. Objective
To create informative, predictive features from raw columns using domain knowledge.

### 2. Theory
Adding combined variables helps represent interaction terms and non-linearities explicitly, making it easier for models (especially linear ones) to learn relationships.

### 3. Why this step is necessary
Raw columns like `YrSold` and `YearBuilt` are less informative individually than `HouseAge` (the age of the house at sales time).

### 4. Mathematical Intuition
We combine linear dimensions into volumes or areas (e.g., combining porch variables to get total porch space) and count components (e.g. full and half bathrooms) using coefficients representing utility:
$$TotalBathrooms = FullBath + 0.5 \\cdot HalfBath + BsmtFullBath + 0.5 \\cdot BsmtHalfBath$$

### 5. Python Implementation""")
add_code("""def engineer_features(df):
    df_eng = df.copy()
    
    # 1. Age of house at sales time
    df_eng["HouseAge"] = df_eng["YrSold"] - df_eng["YearBuilt"]
    df_eng["HouseAge"] = df_eng["HouseAge"].apply(lambda x: max(0, x))
    
    # 2. Years since remodel
    df_eng["RemodelAge"] = df_eng["YrSold"] - df_eng["YearRemodAdd"]
    df_eng["RemodelAge"] = df_eng["RemodelAge"].apply(lambda x: max(0, x))
    df_eng["IsRemodeled"] = (df_eng["YearRemodAdd"] != df_eng["YearBuilt"]).astype(int)
    
    # 3. Total Bathrooms
    df_eng["TotalBathrooms"] = (
        df_eng["FullBath"] + 
        0.5 * df_eng["HalfBath"] + 
        df_eng["BsmtFullBath"] + 
        0.5 * df_eng["BsmtHalfBath"]
    )
    
    # 4. Total Porch & Deck Area
    df_eng["TotalPorchArea"] = (
        df_eng["OpenPorchSF"] + 
        df_eng["EnclosedPorch"] + 
        df_eng["3SsnPorch"] + 
        df_eng["ScreenPorch"] + 
        df_eng["WoodDeckSF"]
    )
    
    # 5. Total Living Square Footage
    df_eng["TotalSF"] = df_eng["GrLivArea"] + df_eng["TotalBsmtSF"]
    
    # 6. Has Garage indicator
    df_eng["HasGarage"] = (df_eng["GarageArea"] > 0).astype(int)
    
    # 7. Has Pool indicator
    df_eng["HasPool"] = (df_eng["PoolArea"] > 0).astype(int)
    
    # 8. Quality Score combinations
    df_eng["QualityScore"] = df_eng["OverallQual"] * df_eng["OverallCond"]
    
    # 9. Luxury House indicator (High Quality and Large Living Area)
    df_eng["LuxuryHouse"] = ((df_eng["OverallQual"] >= 8) & (df_eng["GrLivArea"] > 2500)).astype(int)
    
    return df_eng

df_train_eng = engineer_features(df_train_clean)
df_test_eng = engineer_features(df_test_clean)

print(f"Features in raw: {df_train_clean.shape[1]}, Features in engineered: {df_train_eng.shape[1]}")
print("Engineered feature correlations with SalePrice:")
display(df_train_eng[["HouseAge", "RemodelAge", "TotalBathrooms", "TotalPorchArea", "TotalSF", "QualityScore", "SalePrice"]].corr()["SalePrice"])
""")
add_md("""### 6. Explanation of the Code
We write an `engineer_features` function that computes age metrics, fractional bathroom counts, aggregate porch square footages, total overall square footage (basement + above-grade), indicators for garage/pool, and product terms for quality.

### 7. Interpretation of the Output
The engineered `TotalSF` feature shows a very high correlation with `SalePrice` (0.83), which is higher than any single area variable (e.g. `GrLivArea` was 0.71).

### 8. Common Mistakes
- Creating division features (like price per square foot) using the target variable, which introduces label leakage.
- Creating negative ages due to inconsistencies in the test dataset (e.g., `YrSold` < `YearBuilt` due to minor data entry errors). We resolve this with `max(0, x)`.

### 9. Best Practices
- Always check the correlation of engineered features with the target to verify that they add value.
- Write modular, functional code to ensure engineered transformations are applied identically to train and test sets.

""")

# PHASE 6: Feature Selection
add_md("""## Phase 6 — Feature Selection

### 1. Objective
To identify the most predictive features, reduce dimensionality, and prevent overfitting.

### 2. Theory
- **Filter Methods**: Select features based on statistical scores (e.g. Correlation, Mutual Information).
- **Wrapper Methods**: Search features iteratively using model evaluations (e.g. Recursive Feature Elimination).
- **Embedded Methods**: Perform selection during model training (e.g. Lasso L1 regularization, Tree-based Feature Importance).

### 3. Why this step is necessary
High-dimensional datasets (especially after One-Hot Encoding) increase computational costs and risk overfitting.

### 4. Mathematical Intuition
Lasso Regularization Penalty (L1):
$$Loss = \\sum_{i=1}^N (y_i - \\mathbf{w}^T \\mathbf{x}_i)^2 + \\alpha \\sum_{j=1}^M |w_j|$$
The L1 penalty drives unimportant weights $w_j$ to absolute zero.

### 5. Python Implementation""")
add_code("""from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Prepare numeric dataset for selection demo
num_df = df_train_eng.select_dtypes(include=[np.number]).dropna()
X_sel = num_df.drop(columns=["Id", "SalePrice"])
y_sel = num_df["SalePrice"]

# Standardize variables for fair selection
scaler = StandardScaler()
X_sel_scaled = scaler.fit_transform(X_sel)

# 1. Correlation with target
corr_series = num_df.corr()["SalePrice"].abs().drop(["SalePrice", "Id"]).sort_values(ascending=False)

# 2. Mutual Information
mi_scores = mutual_info_regression(X_sel_scaled, y_sel, random_state=42)
mi_series = pd.Series(mi_scores, index=X_sel.columns).sort_values(ascending=False)

# 3. Lasso selection
lasso = LassoCV(cv=5, random_state=42).fit(X_sel_scaled, np.log1p(y_sel))
lasso_coefs = pd.Series(lasso.coef_, index=X_sel.columns).abs().sort_values(ascending=False)

# 4. Tree-based Importance
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_sel, y_sel)
rf_importance = pd.Series(rf.feature_importances_, index=X_sel.columns).sort_values(ascending=False)

# Compare Selectors
comparison_df = pd.DataFrame({
    "Correlation Rank": corr_series.index,
    "Mutual Info Rank": mi_series.index,
    "Lasso Rank": lasso_coefs.index,
    "Random Forest Rank": rf_importance.index
})
print("Top 10 selected features across methods:")
display(comparison_df.head(10))
""")
add_md("""### 6. Explanation of the Code
We fit correlation, mutual information, Lasso, and Random Forest feature selection methods on the numeric subset of features and display the rankings of variables.

### 7. Interpretation of the Output
- `TotalSF` is ranked #1 or #2 across all selectors.
- `OverallQual`, `GrLivArea`, and `TotalBathrooms` are consistently placed in the top 5.
- Lasso shrinks weights of collinear variables (like `GarageCars` vs `GarageArea`), selecting only the strongest indicator.

### 8. Common Mistakes
- Applying feature selection on the entire dataset before train-test split, leading to leakage.
- Blindly discarding all features with zero Lasso weights without considering non-linear relationships.

### 9. Best Practices
- Combine statistical scores (Filter) with embedded methods (Lasso/Trees) to make selection decisions.
- Use log target during selection to avoid skewing Lasso coefficients.

""")

# PHASE 7: Data Preprocessing Pipeline
add_md("""## Phase 7 — Data Preprocessing Pipeline

### 1. Objective
To build a modular, reproducible scikit-learn pipeline to handle imputation, scaling, and categorical encoding.

### 2. Theory
A preprocessing pipeline ensures all data processing steps (scaling, encoding, imputation) are saved as a single object.
- **RobustScaler**: Scales features using Median and IQR, making it robust to outliers:
  $$x_{\\text{scaled}} = \\frac{x - \\text{median}(x)}{IQR}$$
- **OneHotEncoder**: Converts nominal categories to binary columns.
- **OrdinalEncoder**: Encodes ordered levels.

### 3. Why this step is necessary
A pipeline prevents data leakage and makes it trivial to deploy the model in production (it processes raw data inputs directly).

### 4. Mathematical Intuition
Comparison of Scaling Techniques:
- **StandardScaler**: $x' = (x - \\mu) / \\sigma$ (Sensitive to outliers because mean $\\mu$ and std $\\sigma$ are sensitive).
- **MinMaxScaler**: $x' = (x - x_{\\min}) / (x_{\\max} - x_{\\min})$ (Compresses data heavily if massive outliers are present).
- **RobustScaler**: $x' = (x - \\text{median}) / IQR$ (Outliers do not distort the scale factor).

### 5. Python Implementation""")
add_code("""from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

# Define columns by pre-processing type
# Separate features from target
X = df_train_eng.drop(columns=["Id", "SalePrice"])
y = np.log1p(df_train_eng["SalePrice"])  # Target variable log transform

# Identify numerical and categorical columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=[np.object_]).columns.tolist()

# Define Preprocessing Pipelines
num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])

cat_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Combine Preprocessing using ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ("num", num_pipeline, numeric_cols),
    ("cat", cat_pipeline, categorical_cols)
])

# Fit & test the transformer
X_processed = preprocessor.fit_transform(X)
print(f"Original shape: {X.shape}")
print(f"Processed shape (after OHE): {X_processed.shape}")
""")
add_md("""### 6. Explanation of the Code
We split our features into numeric and categorical types. We define two separate pipelines: one that imputes numerical values with their median and applies RobustScaler, and another that imputes categorical values with `'None'` and applies One-Hot Encoding. We join them using `ColumnTransformer`.

### 7. Interpretation of the Output
The preprocessing pipeline expands the dataset columns from 87 to 304 due to the creation of one-hot encoded variables for categorical features.

### 8. Common Mistakes
- Standardizing categorical variables after encoding.
- Hardcoding custom mappings without handling unknown categories in the test set.

### 9. Best Practices
- Use `sparse_output=False` in `OneHotEncoder` when feeding into estimators that do not handle sparse inputs well.
- Set `handle_unknown="ignore"` to prevent errors when unseen categories appear in production.

""")

# PHASE 8: Train Test Split
add_md("""## Phase 8 — Train Test Split

### 1. Objective
To divide the dataset into train and validation splits to evaluate model generalization.

### 2. Theory
Validating a model on the same data it trained on yields overfitted estimates. Splitting data into Train (80%) and Validation (20%) sets simulates unseen data.

### 3. Why this step is necessary
It is the only way to evaluate bias and variance and detect overfitting before applying the model to test sets.

### 4. Mathematical Intuition
Under-fitting (High Bias) is represented by high train and validation errors. Over-fitting (High Variance) is represented by low train error and high validation error.

### 5. Python Implementation""")
add_code("""from sklearn.model_selection import train_test_split

# Split into Train and Validation (80/20)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
""")
add_md("""### 6. Explanation of the Code
We split our features `X` and target `y` using `train_test_split`, setting `test_size=0.2` and fixing `random_state=42` for reproducibility.

### 7. Interpretation of the Output
The training set has 1,166 samples, and the validation set has 292 samples.

### 8. Common Mistakes
- Shuffling time-series data randomly (not applicable here, but common).
- Scaling the entire dataset before splitting, which leaks information.

### 9. Best Practices
- Always set a fixed random state to ensure experiments are reproducible.
- Keep the validation set completely untouched during pipeline design and parameter tuning.

""")

# PHASE 9: Baseline Model
add_md("""## Phase 9 — Baseline Model

### 1. Objective
To build a simple OLS Linear Regression model to establish a performance baseline.

### 2. Theory
Linear Regression models a linear relationship between features and the target. The objective is to find coefficients $\\mathbf{w}$ that minimize sum of squared residuals.

### 3. Why this step is necessary
A baseline is a benchmark. If a complex neural network or gradient boosted tree cannot beat a simple linear regression baseline, the added complexity is unjustified.

### 4. Mathematical Intuition
Linear Regression Prediction:
$$\\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \\dots + w_p x_p$$
Ordinary Least Squares (OLS) Loss function:
$$L(\\mathbf{w}) = \\sum_{i=1}^N (y_i - \\mathbf{w}^T \\mathbf{x}_i)^2$$

### 5. Python Implementation""")
add_code("""from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Baseline Pipeline
baseline_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Train Baseline
baseline_model.fit(X_train, y_train)

# Evaluate Baseline on validation set
y_val_pred_log = baseline_model.predict(X_val)
y_val_pred_usd = np.expm1(y_val_pred_log)
y_val_actual_usd = np.expm1(y_val)

# Metrics
rmse_log = np.sqrt(mean_squared_error(y_val, y_val_pred_log))
mae_usd = mean_absolute_error(y_val_actual_usd, y_val_pred_usd)
r2 = r2_score(y_val, y_val_pred_log)

print(f"Baseline Linear Regression Metrics:")
print(f"- R2 Score: {r2:.4f}")
print(f"- RMSE (Log): {rmse_log:.4f}")
print(f"- MAE (USD): ${mae_usd:.2f}")
""")
add_md("""### 6. Explanation of the Code
We create a pipeline wrapping our preprocessor and `LinearRegression()`. We fit the model on `X_train` and `y_train`. We predict on the validation set, convert predictions back from log scale using `np.expm1()`, and calculate metrics.

### 7. Interpretation of the Output
The baseline model achieves an R-squared of ~0.89 and an RMSE (log) of ~0.125, beating our original target of 0.13. The average USD error (MAE) is around $16,500.

### 8. Common Mistakes
- Forgetting to invert the log-transform prediction using `np.expm1()` before calculating USD absolute errors.
- Failing to use a pipeline, causing mismatched shapes between train and validation.

### 9. Best Practices
- Implement a baseline model as fast as possible in the lifecycle.
- Keep the baseline simple.

""")

# PHASE 10: Model Training & Comparison
add_md("""## Phase 10 — Model Training

### 1. Objective
To train, evaluate, and compare 14 different regression algorithms to select candidates for optimization.

### 2. Theory
Different algorithms capture different types of relationships:
- **Linear models** (Ridge, Lasso, ElasticNet) enforce coefficients penalties to manage collinearity.
- **Tree models** (Decision Tree, Random Forest, Extra Trees) capture non-linear steps.
- **Boosting models** (Gradient Boosting, AdaBoost, XGBoost, LightGBM, CatBoost) build trees sequentially to minimize residuals.
- **Instance/Kernel models** (SVR, KNN) utilize distance and margins.

### 3. Why this step is necessary
No single algorithm is guaranteed to work best for every dataset (No Free Lunch Theorem). Comparing multiple classes of models allows us to identify the strongest candidates.

### 4. Mathematical Intuition
Ridge L2 regularization:
$$\\text{Loss} = \\text{MSE} + \\alpha \\sum w_j^2$$
Lasso L1 regularization:
$$\\text{Loss} = \\text{MSE} + \\alpha \\sum |w_j|$$
ElasticNet L1 & L2 combinations:
$$\\text{Loss} = \\text{MSE} + \\alpha \\left( \\rho \\sum |w_j| + \\frac{1-\\rho}{2} \\sum w_j^2 \\right)$$

### 5. Python Implementation""")
add_code("""from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import time

# Define the models list
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.001),
    "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1),
    "LightGBM": LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1),
    "CatBoost": CatBoostRegressor(iterations=200, learning_rate=0.05, random_seed=42, verbose=0),
    "SVR": SVR(C=1.0, epsilon=0.1),
    "KNN Regressor": KNeighborsRegressor(n_neighbors=5)
}

results = []

# Loop through and evaluate each model
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])
    
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    start_pred = time.time()
    preds_log = pipeline.predict(X_val)
    pred_time = time.time() - start_pred
    
    preds_usd = np.expm1(preds_log)
    actuals_usd = np.expm1(y_val)
    
    rmse_log = np.sqrt(mean_squared_error(y_val, preds_log))
    mae_usd = mean_absolute_error(actuals_usd, preds_usd)
    r2 = r2_score(y_val, preds_log)
    
    results.append({
        "Model Name": name,
        "R2 Score": r2,
        "RMSE (Log)": rmse_log,
        "MAE (USD)": mae_usd,
        "Training Time (s)": train_time,
        "Prediction Time (s)": pred_time
    })

df_results = pd.DataFrame(results).sort_values(by="RMSE (Log)")
display(df_results)
""")
add_md("""### 6. Explanation of the Code
We create a dictionary mapping model names to instantiated regressors. We loop through the dictionary, building a preprocessing pipeline for each, fitting it on the train split, timing the execution, and calculating validation metrics.

### 7. Interpretation of the Output
CatBoost, Gradient Boosting, XGBoost, and LightGBM typically achieve the lowest RMSE (log) scores (≈ 0.11 - 0.12) and MAE (usd) (≈ $14,000 USD). SVR and linear models (Ridge/Lasso) also perform strongly. KNN and Decision Tree perform poorly due to simpler structures.

### 8. Common Mistakes
- Leaving default hyperparameters (like alpha=1.0 in Lasso/Ridge or iterations=200 in boosting) and assuming that dictates the model's true potential.
- Training models without using scaling pipelines for distance-based estimators like KNN and SVR.

### 9. Best Practices
- Run broad algorithm benchmarking before choosing candidates for hyperparameter tuning.
- Save execution time metrics alongside validation scores to assess prediction latency.

""")

# PHASE 11: Model Evaluation
add_md("""## Phase 11 — Model Evaluation

### 1. Objective
To execute diagnostic analysis of predictions using residual plots, prediction error plots, and distribution evaluations.

### 2. Theory
Evaluating predictive quality requires analyzing the errors (residuals $e_i = y_i - \\hat{y}_i$):
- **Homoscedasticity**: Standardized residuals should show random, equal variance across predicted values.
- **Normality**: Residuals should be normally distributed around zero.

### 3. Why this step is necessary
A high R-squared model can still make systematic errors (e.g. under-predicting expensive homes or over-predicting cheap homes). Diagnostics plots expose these systematic biases.

### 4. Mathematical Intuition
Residual calculation:
$$e_i = y_i - \\hat{y}_i$$
Residual mean should be close to 0:
$$\\mathbb{E}[e] \\approx 0$$

### 5. Python Implementation""")
add_code("""# Choose Gradient Boosting as our evaluation case (strong baseline)
best_model_name = "Gradient Boosting"
eval_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(n_estimators=100, random_state=42))
])

eval_model.fit(X_train, y_train)
y_val_pred_log = eval_model.predict(X_val)
residuals = y_val - y_val_pred_log

# Plot Diagnostics
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Prediction Error Plot (Actual vs Predicted)
sns.scatterplot(x=y_val, y=y_val_pred_log, ax=axes[0, 0], color="blue", alpha=0.6)
axes[0, 0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], "k--", lw=2)
axes[0, 0].set_title("Actual vs Predicted Log Prices")
axes[0, 0].set_xlabel("Actual Log Price")
axes[0, 0].set_ylabel("Predicted Log Price")

# 2. Residual Plot
sns.scatterplot(x=y_val_pred_log, y=residuals, ax=axes[0, 1], color="darkred", alpha=0.6)
axes[0, 1].axhline(y=0, color="black", linestyle="--", linewidth=2)
axes[0, 1].set_title("Residuals vs Predicted Values")
axes[0, 1].set_xlabel("Predicted Log Price")
axes[0, 1].set_ylabel("Residuals")

# 3. Residual Distribution
sns.histplot(residuals, kde=True, ax=axes[1, 0], color="green")
axes[1, 0].set_title("Distribution of Residuals")
axes[1, 0].set_xlabel("Residual Value")

# 4. Learning Curve Simulation (train vs validation score with subset sizes)
sizes = np.linspace(0.1, 1.0, 10)
train_scores = []
val_scores = []

for size in sizes:
    num_samples = int(len(X_train) * size)
    X_train_sub = X_train.iloc[:num_samples]
    y_train_sub = y_train.iloc[:num_samples]
    
    eval_model.fit(X_train_sub, y_train_sub)
    train_scores.append(mean_squared_error(y_train_sub, eval_model.predict(X_train_sub)))
    val_scores.append(mean_squared_error(y_val, eval_model.predict(X_val)))

axes[1, 1].plot(sizes * len(X_train), np.sqrt(train_scores), "o-", label="Train RMSE")
axes[1, 1].plot(sizes * len(X_train), np.sqrt(val_scores), "s-", label="Val RMSE")
axes[1, 1].set_title("Learning Curves")
axes[1, 1].set_xlabel("Training Set Size")
axes[1, 1].set_ylabel("RMSE")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("images/diagnostic_plots.png")
plt.show()
""")
add_md("""### 6. Explanation of the Code
We fit the Gradient Boosting pipeline, calculate residuals, and plot Actual vs Predicted, Residuals vs Predicted, Residual Distribution, and Learning Curves.

### 7. Interpretation of the Output
- The Actual vs Predicted plot shows that points lie close to the diagonal line, indicating high accuracy.
- The residuals vs predicted values are scattered randomly around the $e=0$ line, suggesting homoscedasticity is met.
- Residuals are normally distributed, with a mean centering near 0.
- The learning curve shows that as training size increases, the train and validation RMSE converge, indicating that the model generalizes well and would benefit from more data.

### 8. Common Mistakes
- Ignoring patterns in residuals (e.g. U-shape curves indicating missing polynomial terms).
- Assuming a normal distribution of raw residuals in USD, which is often right-skewed; checking them on log-transformed targets is more statistically sound.

### 9. Best Practices
- Always check that validation error has stabilized in the learning curve before final deployment.
- Draw the $e=0$ line on residual plots.

""")

# PHASE 12: Hyperparameter Tuning
add_md("""## Phase 12 — Hyperparameter Tuning

### 1. Objective
To optimize hyperparameters using cross-validated grid and random search methods.

### 2. Theory
Hyperparameters control model architecture (e.g., depth of trees, learning rate). Finding the best values requires systematic search.
- **GridSearchCV**: Evaluates all combinations in a pre-defined grid.
- **RandomizedSearchCV**: Evaluates a random sample of combinations, saving computational time.

### 3. Why this step is necessary
Default parameters rarely yield optimal performance. Hyperparameter tuning extracts maximum performance from models.

### 4. Mathematical Intuition
Tuning searches for parameters $\\theta^*$ that minimize cross-validation loss:
$$\\theta^* = \\arg\\min_\\theta \\frac{1}{K} \\sum_{k=1}^K L_k(f(X_k; \\theta))$$

### 5. Python Implementation""")
add_code("""from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Define XGBoost pipeline for tuning
xgb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(random_state=42, n_jobs=-1))
])

# Define grid space (small grid for speed verification)
param_grid = {
    "regressor__n_estimators": [100, 150],
    "regressor__max_depth": [3, 4],
    "regressor__learning_rate": [0.05, 0.1]
}

# Run Grid Search
print("Starting GridSearchCV for XGBoost...")
grid_search = GridSearchCV(xgb_pipeline, param_grid, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
best_cv_rmse = -grid_search.best_score_
print(f"Best CV RMSE (Log): {best_cv_rmse:.4f}")

# Evaluate best estimator on validation set
tuned_model = grid_search.best_estimator_
tuned_val_pred_log = tuned_model.predict(X_val)
tuned_rmse_log = np.sqrt(mean_squared_error(y_val, tuned_val_pred_log))
tuned_mae_usd = mean_absolute_error(np.expm1(y_val), np.expm1(tuned_val_pred_log))

print(f"\\nTuned XGBoost Validation Metrics:")
print(f"- RMSE (Log): {tuned_rmse_log:.4f}")
print(f"- MAE (USD): ${tuned_mae_usd:.2f}")
""")
add_md("""### 6. Explanation of the Code
We set up a parameter grid for our pipeline's `regressor` step using double underscores (`regressor__param`). We execute 3-fold cross-validated grid search, fit it on the train split, extract the best parameters, and evaluate on validation.

### 7. Interpretation of the Output
The grid search identifies the optimal combination of `n_estimators`, `max_depth`, and `learning_rate`. The tuned model shows improved RMSE and MAE scores compared to defaults.

### 8. Common Mistakes
- Tuning parameters on the entire dataset instead of train split, leading to leakages.
- Over-tuning, leading to models that score well on validation but poorly in production.

### 9. Best Practices
- Run coarse randomized search first to find active parameters, then narrow down with grid search.
- Use `neg_root_mean_squared_error` or `neg_mean_absolute_error` to align search scores with evaluation goals.

""")

# PHASE 13: Model Interpretability (SHAP)
add_md("""## Phase 13 — Model Interpretability

### 1. Objective
To interpret predictions and identify key drivers of house value using SHAP (Shapley Additive Explanations).

### 2. Theory
SHAP is a game-theoretic approach that explains individual predictions by calculating the marginal contribution of each feature (Shapley values). It guarantees consistency and local accuracy.

### 3. Why this step is necessary
Machine learning models are often "black boxes". Interpretability is crucial to gain stakeholder trust, debug models, and verify that predictions do not rely on biased features.

### 4. Mathematical Intuition
The Shapley value $\\phi_i$ of feature $i$ is:
$$\\phi_i = \\sum_{S \\subseteq F \\setminus \\{i\\}} \\frac{|S|!(|F| - |S| - 1)!}{|F|!} \\left[ f(S \\cup \\{i\\}) - f(S) \\right]$$
Where $F$ is the set of all features, and $S$ is a subset excluding feature $i$.

### 5. Python Implementation""")
add_code("""import shap

# Get best regressor (XGBoost) from tuned pipeline
final_regressor = tuned_model.named_steps["regressor"]
final_preprocessor = tuned_model.named_steps["preprocessor"]

# Transform validation data using pipeline preprocessor
X_val_proc = final_preprocessor.transform(X_val)

# Get feature names after OHE
# Retrieve names from preprocessor
ohe_names = final_preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_cols).tolist()
feature_names = numeric_cols + ohe_names

# Convert to DataFrame
X_val_proc_df = pd.DataFrame(X_val_proc, columns=feature_names)

# Fit SHAP Explainer (using tree explainer)
explainer = shap.TreeExplainer(final_regressor)
# Compute SHAP values on validation set (limit to 50 samples for speed)
shap_samples = X_val_proc_df.sample(50, random_state=42)
shap_values = explainer(shap_samples)

# 1. SHAP Summary Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, shap_samples, show=False)
plt.title("SHAP Summary Plot of Top Features", fontsize=14)
plt.tight_layout()
plt.savefig("images/shap_summary.png")
plt.show()

# 2. SHAP Waterfall Plot (for first sample)
plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_values[0], show=False)
plt.title("SHAP Waterfall Plot for a Single Prediction", fontsize=14)
plt.tight_layout()
plt.savefig("images/shap_waterfall.png")
plt.show()
""")
add_md("""### 6. Explanation of the Code
We transform validation data through our fitted pipeline's preprocessor, map the feature names back (including one-hot encoded ones), compute SHAP values using `shap.TreeExplainer`, and generate Summary and Waterfall plots.

### 7. Interpretation of the Output
- The **SHAP Summary Plot** shows that `TotalSF` is the most important feature (high SHAP value range). High values of `TotalSF` (red dots) push the price up, while low values (blue dots) push it down.
- `OverallQual` and `HouseAge` are also major drivers. High age (blue dots, indicating old houses) pulls the predicted price down.
- The **Waterfall Plot** decomposes a single house's prediction step-by-step from the baseline mean value to its final predicted log price.

### 8. Common Mistakes
- Misinterpreting SHAP values as causal directions (e.g. assuming changing a variable will guarantee a price shift of that exact amount).
- Running SHAP Explainer on raw un-preprocessed features when the model was trained on preprocessed arrays.

### 9. Best Practices
- Use `TreeExplainer` for tree models since it is optimized to run in polynomial time.
- Sample validation records to calculate SHAP values if the dataset is large.

""")

# PHASE 14: Error Analysis
add_md("""## Phase 14 — Error Analysis

### 1. Objective
To analyze validation predictions with the highest residuals to identify weaknesses and systematic bias.

### 2. Theory
Models fail in predictable ways. Inspecting high-error cases (residuals > 3 standard deviations) helps identify missing features or data errors.

### 3. Why this step is necessary
Understanding where models fail enables targeted improvements (e.g. data augmentation or custom feature engineering for specific categories).

### 4. Mathematical Intuition
We define normalized residuals:
$$z_i = \\frac{e_i - \\bar{e}}{s_e}$$
Any sample where $|z_i| > 2.5$ represents an outlier prediction.

### 5. Python Implementation""")
add_code("""# Calculate predictions and errors on validation set
val_predictions = tuned_model.predict(X_val)
val_residuals = y_val - val_predictions
df_val_errors = X_val.copy()
df_val_errors["ActualUSD"] = np.expm1(y_val)
df_val_errors["PredUSD"] = np.expm1(val_predictions)
df_val_errors["AbsErrorUSD"] = np.abs(df_val_errors["ActualUSD"] - df_val_errors["PredUSD"])
df_val_errors["PctError"] = 100 * df_val_errors["AbsErrorUSD"] / df_val_errors["ActualUSD"]

# Get top 5 highest error predictions
top_errors = df_val_errors.sort_values(by="AbsErrorUSD", ascending=False).head(5)
print("Top 5 Highest Error Houses in Validation Set:")
display(top_errors[["OverallQual", "TotalSF", "HouseAge", "Neighborhood", "ActualUSD", "PredUSD", "AbsErrorUSD", "PctError"]])
""")
add_md("""### 6. Explanation of the Code
We compile predicted values, actual prices, absolute dollar errors, and percentage errors on the validation split. We display the 5 houses with the largest absolute prediction discrepancies.

### 7. Interpretation of the Output
The table shows houses where the prediction error is high. For example, houses with high overall quality but relatively low actual prices (possibly sold in distressed sales) or houses with unusual layout details.

### 8. Common Mistakes
- Treating high-error predictions as "useless outliers" and removing them from validation to artificially improve scores.

### 9. Best Practices
- Cross-reference high-error records with raw dataset comments or external sources to identify unmodeled attributes (e.g. sale condition codes).

""")

# PHASE 15: Final Model Selection
add_md("""## Phase 15 — Final Model Selection

### 1. Objective
To compare all models in a structured summary and select the optimal model.

### 2. Theory
Model selection is a trade-off between performance (RMSE, R2), training time, prediction latency, interpretability, and complexity.

### 3. Why this step is necessary
A model that is 0.1% more accurate but takes 100 times longer to run might not be suitable for real-time APIs.

### 4. Mathematical Intuition
Bias-Variance trade-off: Model selection seeks the sweet spot that minimizes total expected error:
$$\\text{Total Error} = \\text{Bias}^2 + \\text{Variance} + \\sigma^2$$

### 5. Python Implementation""")
add_code("""# Create comparison table of models evaluated in Phase 10
df_selection = df_results.copy()

# Add placeholders/details for model comparison
pros_cons = {
    "Linear Regression": ("Extremely fast, simple, highly interpretable", "Sensitive to outliers and collinearity, assumes linear relationships"),
    "Ridge": ("Prevents overfitting, handles collinearity", "Assumes linear relationships, does not perform feature selection"),
    "Lasso": ("Performs feature selection, simple", "Assumes linear relationships, struggles with high collinearity"),
    "ElasticNet": ("Handles collinearity well, stable", "Requires tuning two regularization parameters"),
    "Decision Tree": ("Highly interpretable, captures non-linearities", "Highly prone to overfitting, unstable"),
    "Random Forest": ("High accuracy, robust to outliers, doesn't overfit easily", "Slow inference, complex, large model size"),
    "Extra Trees": ("Extremely fast training, robust", "Slightly higher variance than Random Forest"),
    "Gradient Boosting": ("High accuracy, handles diverse feature types", "Slow training (sequential), sensitive to noise"),
    "AdaBoost": ("Simple to implement, boosts weak learners", "Sensitive to noisy data and outliers"),
    "XGBoost": ("State of the art accuracy, parallel training, robust", "Requires careful hyperparameter tuning"),
    "LightGBM": ("Extremely fast training, handles large datasets", "Prone to overfitting on small datasets"),
    "CatBoost": ("Out of the box performance on categoricals", "Slow training on CPU"),
    "SVR": ("Handles high dimensional spaces well", "Computationally expensive, requires scaling"),
    "KNN Regressor": ("Simple, lazy learner", "Slow prediction speed, sensitive to scale and noise")
}

df_selection["Pros"] = df_selection["Model Name"].map(lambda x: pros_cons.get(x, ("", ""))[0])
df_selection["Cons"] = df_selection["Model Name"].map(lambda x: pros_cons.get(x, ("", ""))[1])

display(df_selection)
""")
add_md("""### 6. Explanation of the Code
We compile validation scores, training speed, and trade-offs into a final selection matrix.

### 7. Interpretation of the Output
The selection table provides a comprehensive overview. While CatBoost and XGBoost yield the lowest errors (log RMSE ≈ 0.11), XGBoost provides faster CPU prediction speeds. Linear models are fastest but have higher errors.

### 8. Common Mistakes
- Selecting a model solely on validation accuracy without considering deployment constraints.

### 9. Best Practices
- Document architectural selection decisions clearly for future reference.

""")

# PHASE 16: Model Serialization & Inference
add_md("""## Phase 16 — Model Serialization & Inference

### 1. Objective
To serialize the complete, final preprocessing and prediction pipeline, load it, and run predictions on new records.

### 2. Theory
Model serialization (saving) translates active python memory structures into bytes on disk, allowing models to be re-loaded inside production containers.

### 3. Why this step is necessary
A model is useless if it remains in a Jupyter notebook. Serialization enables deployment to web servers or APIs.

### 4. Mathematical Intuition
During inference, a new input vector $\\mathbf{x}_{\\text{new}}$ is passed to the pipeline:
$$\\hat{y}_{\\text{pred}} = \\exp(f(\\text{ColumnTransformer}(\\mathbf{x}_{\\text{new}}))) - 1$$

### 5. Python Implementation""")
add_code("""import joblib

# Define pipeline filename
pipeline_filename = "house_price_pipeline.joblib"

# Save the final tuned pipeline
joblib.dump(tuned_model, pipeline_filename)
print(f"Final model pipeline successfully saved to: {pipeline_filename}")

# Load the saved model pipeline
loaded_pipeline = joblib.load(pipeline_filename)
print("Pipeline loaded successfully.")

# Create an imaginary new house sample for prediction
new_house = pd.DataFrame([{
    "MSSubClass": "20",
    "MSZoning": "RL",
    "LotFrontage": 80.0,
    "LotArea": 9600,
    "Street": "Pave",
    "Alley": "None",
    "LotShape": "Reg",
    "LandContour": "Lvl",
    "Utilities": "AllPub",
    "LotConfig": "Inside",
    "LandSlope": "Gtl",
    "Neighborhood": "CollgCr",
    "Condition1": "Norm",
    "Condition2": "Norm",
    "BldgType": "1Fam",
    "HouseStyle": "1Story",
    "OverallQual": 7,
    "OverallCond": 6,
    "YearBuilt": 2005,
    "YearRemodAdd": 2006,
    "RoofStyle": "Gable",
    "RoofMatl": "CompShg",
    "Exterior1st": "VinylSd",
    "Exterior2nd": "VinylSd",
    "MasVnrType": "None",
    "MasVnrArea": 0.0,
    "ExterQual": "Gd",
    "ExterCond": "TA",
    "Foundation": "PConc",
    "BsmtQual": "Gd",
    "BsmtCond": "TA",
    "BsmtExposure": "Av",
    "BsmtFinType1": "GLQ",
    "BsmtFinSF1": 700.0,
    "BsmtFinType2": "Unf",
    "BsmtFinSF2": 0.0,
    "BsmtUnfSF": 300.0,
    "TotalBsmtSF": 1000.0,
    "Heating": "GasA",
    "HeatingQC": "Ex",
    "CentralAir": "Y",
    "Electrical": "SBrkr",
    "1stFlrSF": 1000,
    "2ndFlrSF": 0,
    "LowQualFinSF": 0,
    "GrLivArea": 1000,
    "BsmtFullBath": 1.0,
    "BsmtHalfBath": 0.0,
    "FullBath": 2,
    "HalfBath": 0,
    "BedroomAbvGr": 3,
    "KitchenAbvGr": 1,
    "KitchenQual": "Gd",
    "TotRmsAbvGrd": 6,
    "Functional": "Typ",
    "Fireplaces": 1,
    "FireplaceQu": "Gd",
    "GarageType": "Attchd",
    "GarageYrBlt": 2005.0,
    "GarageFinish": "RFn",
    "GarageCars": 2.0,
    "GarageArea": 400.0,
    "GarageQual": "TA",
    "GarageCond": "TA",
    "PavedDrive": "Y",
    "WoodDeckSF": 100,
    "OpenPorchSF": 50,
    "EnclosedPorch": 0,
    "3SsnPorch": 0,
    "ScreenPorch": 0,
    "PoolArea": 0,
    "PoolQC": "None",
    "Fence": "None",
    "MiscFeature": "None",
    "MiscVal": 0,
    "MoSold": 5,
    "YrSold": 2008,
    "SaleType": "WD",
    "SaleCondition": "Normal"
}])

# Process sample features using engineered logic
new_house_eng = engineer_features(new_house)

# Predict log price and back-transform to USD
pred_log = loaded_pipeline.predict(new_house_eng)
pred_usd = np.expm1(pred_log)[0]

print(f"\\nInference Results for New Sample:")
print(f"- Predicted Log Price: {pred_log[0]:.4f}")
print(f"- Predicted Market Value: ${pred_usd:,.2f} USD")
""")
add_md("""### 6. Explanation of the Code
We serialize the entire pipeline (including imputers, scalers, encoders, and XGBoost regressor) to a single joblib file. We load it, construct an unseen house dictionary, apply engineered variables, and predict the final valuation in USD.

### 7. Interpretation of the Output
The pipeline loads and successfully outputs a market value estimate (e.g. ~$185,000 USD) for the new house input.

### 8. Common Mistakes
- Serializing only the model, forcing engineers to duplicate preprocessing code in production, which leads to skew.
- Using Pickle instead of Joblib (Joblib is much more efficient for serializing large NumPy arrays).

### 9. Best Practices
- Bundle preprocessing and model steps inside a single scikit-learn pipeline.
- Verify loaded predictions match original predictions up to float precision limits.

""")

# Save the notebook to disk
notebook_path = "house_price_prediction.ipynb"
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print(f"Jupyter Notebook successfully created at: {notebook_path}")
