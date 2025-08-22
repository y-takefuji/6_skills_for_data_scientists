import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import scipy.stats as stats
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer

# Load the data
data = pd.read_csv('BKB_WaterQualityData_2020084.csv')
print(data.shape)
# Separate numeric and non-numeric columns
numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
non_numeric_columns = data.select_dtypes(exclude=np.number).columns.tolist()

print(f"Numeric columns: {numeric_columns}")
print(f"Non-numeric columns: {non_numeric_columns}")

# Handle missing values
# 1. Remove rows with missing values in target or Dissolved Oxygen
data = data.dropna(subset=['Salinity (ppt)', 'Dissolved Oxygen (mg/L)'])

# 2. Encode all categorical columns including Site_Id
for col in non_numeric_columns:
    if col in data.columns:  # Make sure the column exists
        le = LabelEncoder()
        # Handle potential missing values in categorical columns
        if data[col].isna().any():
            # Create a temporary series with NaN replaced by a placeholder
            temp_series = data[col].fillna('MISSING')
            # Fit and transform
            data[col] = le.fit_transform(temp_series)
        else:
            data[col] = le.fit_transform(data[col])

# Now identify columns for imputation (only numeric columns with missing values)
columns_to_impute = [col for col in numeric_columns 
                    if col not in ['Salinity (ppt)', 'Dissolved Oxygen (mg/L)'] 
                    and data[col].isna().any()]

# Impute missing values in numeric columns
if columns_to_impute:
    imputer = SimpleImputer(strategy='mean')
    data[columns_to_impute] = imputer.fit_transform(data[columns_to_impute])

# Define target and features
target = 'Salinity (ppt)'
exclude = [target]
features = [col for col in data.columns if col not in exclude]

X = data[features]
y = data[target]

print(f"Dataset shape after handling missing values: {X.shape}")
print(f"Features being used: {X.columns.tolist()}")

# Define feature selection methods
def random_forest_feature_importance(X, y, k=4):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    selected_features = [X.columns[i] for i in indices[:k]]
    selected_importances = [importances[i] for i in indices[:k]]
    return selected_features, selected_importances

def xgboost_feature_importance(X, y, k=4):
    xgb = XGBRegressor(n_estimators=100, random_state=42)
    xgb.fit(X, y)
    importances = xgb.feature_importances_
    indices = np.argsort(importances)[::-1]
    selected_features = [X.columns[i] for i in indices[:k]]
    selected_importances = [importances[i] for i in indices[:k]]
    return selected_features, selected_importances

def feature_agglomeration(X, k=4):
    fa = FeatureAgglomeration(n_clusters=k)
    fa.fit(X)
    clusters = fa.labels_
    
    # For each cluster, find the feature with highest variance
    selected_features = []
    variances = []
    
    for i in range(k):
        cluster_features = [X.columns[j] for j in range(len(X.columns)) if clusters[j] == i]
        if cluster_features:
            variances_in_cluster = [X[feat].var() for feat in cluster_features]
            max_var_idx = np.argmax(variances_in_cluster)
            selected_features.append(cluster_features[max_var_idx])
            variances.append(variances_in_cluster[max_var_idx])
    
    # Sort by variance
    sorted_indices = np.argsort(variances)[::-1]
    return [selected_features[i] for i in sorted_indices], [variances[i] for i in sorted_indices]

def highly_variable_features(X, k=4):
    variances = X.var().sort_values(ascending=False)
    return list(variances.index[:k]), list(variances.values[:k])

def spearman_correlation(X, y, k=4):
    correlations = []
    p_values = []
    
    for feature in X.columns:
        corr, p_val = spearmanr(X[feature], y)
        correlations.append(abs(corr))  # Use absolute correlation
        p_values.append(p_val)
    
    indices = np.argsort(correlations)[::-1]
    selected_features = [X.columns[i] for i in indices[:k]]
    selected_correlations = [correlations[i] for i in indices[:k]]
    selected_pvalues = [p_values[i] for i in indices[:k]]
    return selected_features, selected_correlations, selected_pvalues

# Evaluate selected features with Random Forest
def evaluate_features(X, y, selected_features):
    X_selected = X[selected_features]
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X_selected, y, cv=cv, scoring='r2')
    return np.mean(scores), np.std(scores)

# Run all feature selection methods and evaluate
print("\nFeature Selection Results (Top 4 Features):")
print("=" * 60)

# 1. Random Forest
rf_features, rf_importances = random_forest_feature_importance(X, y)
rf_score, rf_std = evaluate_features(X, y, rf_features)
print(f"Random Forest: {rf_features}")
print(f"Importances: {rf_importances}")
print(f"Cross-validation R² score: {rf_score:.4f} ± {rf_std:.4f}")
print("-" * 60)

# 2. XGBoost
xgb_features, xgb_importances = xgboost_feature_importance(X, y)
xgb_score, xgb_std = evaluate_features(X, y, xgb_features)
print(f"XGBoost: {xgb_features}")
print(f"Importances: {xgb_importances}")
print(f"Cross-validation R² score: {xgb_score:.4f} ± {xgb_std:.4f}")
print("-" * 60)

# 3. Feature Agglomeration
fa_features, fa_variances = feature_agglomeration(X)
fa_score, fa_std = evaluate_features(X, y, fa_features)
print(f"Feature Agglomeration: {fa_features}")
print(f"Variances: {fa_variances}")
print(f"Cross-validation R² score: {fa_score:.4f} ± {fa_std:.4f}")
print("-" * 60)

# 4. Highly Variable Features
hv_features, hv_variances = highly_variable_features(X)
hv_score, hv_std = evaluate_features(X, y, hv_features)
print(f"Highly Variable Features: {hv_features}")
print(f"Variances: {hv_variances}")
print(f"Cross-validation R² score: {hv_score:.4f} ± {hv_std:.4f}")
print("-" * 60)

# 5. Spearman Correlation
sp_features, sp_correlations, sp_pvalues = spearman_correlation(X, y)
sp_score, sp_std = evaluate_features(X, y, sp_features)
print(f"Spearman Correlation: {sp_features}")
print(f"Correlations: {sp_correlations}")
print(f"P-values: {sp_pvalues}")
print(f"Cross-validation R² score: {sp_score:.4f} ± {sp_std:.4f}")
print("=" * 60)

# Create reduced datasets by removing top feature from each method
print("\nStability Analysis (Top 3 Features from Reduced Dataset):")
print("=" * 60)

# 1. Random Forest - Remove top feature and select top 3
reduced_X_rf = X.drop(columns=[rf_features[0]])
rf_reduced_features, rf_reduced_importances = random_forest_feature_importance(reduced_X_rf, y, k=3)
rf_reduced_score, rf_reduced_std = evaluate_features(reduced_X_rf, y, rf_reduced_features)
print(f"Random Forest (Reduced): {rf_reduced_features}")
print(f"Importances: {rf_reduced_importances}")
print(f"Stability: Original top 4 = {rf_features}, Reduced top 3 = {rf_reduced_features}")
print(f"Overlap with original features: {[f for f in rf_reduced_features if f in rf_features[1:]]}")
print(f"Cross-validation R² score: {rf_reduced_score:.4f} ± {rf_reduced_std:.4f}")
print("-" * 60)

# 2. XGBoost - Remove top feature and select top 3
reduced_X_xgb = X.drop(columns=[xgb_features[0]])
xgb_reduced_features, xgb_reduced_importances = xgboost_feature_importance(reduced_X_xgb, y, k=3)
xgb_reduced_score, xgb_reduced_std = evaluate_features(reduced_X_xgb, y, xgb_reduced_features)
print(f"XGBoost (Reduced): {xgb_reduced_features}")
print(f"Importances: {xgb_reduced_importances}")
print(f"Stability: Original top 4 = {xgb_features}, Reduced top 3 = {xgb_reduced_features}")
print(f"Overlap with original features: {[f for f in xgb_reduced_features if f in xgb_features[1:]]}")
print(f"Cross-validation R² score: {xgb_reduced_score:.4f} ± {xgb_reduced_std:.4f}")
print("-" * 60)

# 3. Feature Agglomeration - Remove top feature and select top 3
reduced_X_fa = X.drop(columns=[fa_features[0]])
fa_reduced_features, fa_reduced_variances = feature_agglomeration(reduced_X_fa, k=3)
fa_reduced_score, fa_reduced_std = evaluate_features(reduced_X_fa, y, fa_reduced_features)
print(f"Feature Agglomeration (Reduced): {fa_reduced_features}")
print(f"Variances: {fa_reduced_variances}")
print(f"Stability: Original top 4 = {fa_features}, Reduced top 3 = {fa_reduced_features}")
print(f"Overlap with original features: {[f for f in fa_reduced_features if f in fa_features[1:]]}")
print(f"Cross-validation R² score: {fa_reduced_score:.4f} ± {fa_reduced_std:.4f}")
print("-" * 60)

# 4. Highly Variable Features - Remove top feature and select top 3
reduced_X_hv = X.drop(columns=[hv_features[0]])
hv_reduced_features, hv_reduced_variances = highly_variable_features(reduced_X_hv, k=3)
hv_reduced_score, hv_reduced_std = evaluate_features(reduced_X_hv, y, hv_reduced_features)
print(f"Highly Variable Features (Reduced): {hv_reduced_features}")
print(f"Variances: {hv_reduced_variances}")
print(f"Stability: Original top 4 = {hv_features}, Reduced top 3 = {hv_reduced_features}")
print(f"Overlap with original features: {[f for f in hv_reduced_features if f in hv_features[1:]]}")
print(f"Cross-validation R² score: {hv_reduced_score:.4f} ± {hv_reduced_std:.4f}")
print("-" * 60)

# 5. Spearman Correlation - Remove top feature and select top 3
reduced_X_sp = X.drop(columns=[sp_features[0]])
sp_reduced_features, sp_reduced_correlations, sp_reduced_pvalues = spearman_correlation(reduced_X_sp, y, k=3)
sp_reduced_score, sp_reduced_std = evaluate_features(reduced_X_sp, y, sp_reduced_features)
print(f"Spearman Correlation (Reduced): {sp_reduced_features}")
print(f"Correlations: {sp_reduced_correlations}")
print(f"P-values: {sp_reduced_pvalues}")
print(f"Stability: Original top 4 = {sp_features}, Reduced top 3 = {sp_reduced_features}")
print(f"Overlap with original features: {[f for f in sp_reduced_features if f in sp_features[1:]]}")
print(f"Cross-validation R² score: {sp_reduced_score:.4f} ± {sp_reduced_std:.4f}")
print("=" * 60)

# Final summary
print("\nFeature Selection Stability Summary:")
print("=" * 60)
all_methods = ["Random Forest", "XGBoost", "Feature Agglomeration", "Highly Variable", "Spearman"]
all_original = [rf_features, xgb_features, fa_features, hv_features, sp_features]
all_reduced = [rf_reduced_features, xgb_reduced_features, fa_reduced_features, hv_reduced_features, sp_reduced_features]

for i, method in enumerate(all_methods):
    overlap = [f for f in all_reduced[i] if f in all_original[i][1:]]
    stability_percentage = (len(overlap) / min(len(all_reduced[i]), len(all_original[i])-1)) * 100
    print(f"{method}: {stability_percentage:.1f}% stability (overlap: {len(overlap)}/{min(len(all_reduced[i]), len(all_original[i])-1)})")

# Find features that appear in multiple methods
print("\nFeature Frequency Across Methods (original top 4):")
feature_freq = {}
for features_list in all_original:
    for feature in features_list:
        feature_freq[feature] = feature_freq.get(feature, 0) + 1

for feature, freq in sorted(feature_freq.items(), key=lambda x: x[1], reverse=True):
    print(f"{feature}: appears in {freq}/{len(all_methods)} methods")

print("\nFeature Frequency Across Methods (reduced top 3):")
feature_freq_reduced = {}
for features_list in all_reduced:
    for feature in features_list:
        feature_freq_reduced[feature] = feature_freq_reduced.get(feature, 0) + 1

for feature, freq in sorted(feature_freq_reduced.items(), key=lambda x: x[1], reverse=True):
    print(f"{feature}: appears in {freq}/{len(all_methods)} methods")
