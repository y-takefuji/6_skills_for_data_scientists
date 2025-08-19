import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import FeatureAgglomeration
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('food_allergy_dataset.csv')

# Separate features and target
X = df.drop(['Allergic'], axis=1)
y = df['Allergic']
print(df.shape)

# Convert categorical variables to one-hot encoding
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
feature_names = X_encoded.columns.tolist()
X_processed = X_encoded.values

# ---- Feature Selection Methods with Tracked Indices ----

# 1. Random Forest Feature Importance
def select_features_rf(X, y, feature_names, n_features=5):
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    selected_indices = indices[:n_features]
    selected_features = [feature_names[i] for i in selected_indices]
    return selected_features, selected_indices

# 2. Logistic Regression Coefficients
def select_features_logistic(X, y, feature_names, n_features=5):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    coefs = np.abs(model.coef_[0])
    indices = np.argsort(coefs)[::-1]
    
    selected_indices = indices[:n_features]
    selected_features = [feature_names[i] for i in selected_indices]
    return selected_features, selected_indices

# 3. Feature Agglomeration
def select_features_agglomeration(X, y, feature_names, n_features=5):
    n_clusters = min(len(feature_names) - n_features, len(feature_names) - 1)
    if n_clusters <= 0:
        selected_indices = np.arange(min(n_features, len(feature_names)))
        return [feature_names[i] for i in selected_indices], selected_indices
        
    agglo = FeatureAgglomeration(n_clusters=n_clusters)
    agglo.fit(X)
    
    # Calculate feature importance based on variance in each cluster
    feature_importances = np.zeros(len(feature_names))
    for i in range(len(feature_names)):
        feature_importances[i] = np.var(X[:, agglo.labels_ == agglo.labels_[i]])
    
    indices = np.argsort(feature_importances)[::-1]
    
    selected_indices = indices[:n_features]
    selected_features = [feature_names[i] for i in selected_indices]
    return selected_features, selected_indices

# 4. Highly Variable Feature Selection
def select_features_variance(X, y, feature_names, n_features=5):
    variances = np.var(X, axis=0)
    indices = np.argsort(variances)[::-1]
    
    selected_indices = indices[:n_features]
    selected_features = [feature_names[i] for i in selected_indices]
    return selected_features, selected_indices

# 5. Spearman's Correlation
def select_features_spearman(X, y, feature_names, n_features=5):
    correlations = []
    
    for i in range(X.shape[1]):
        corr, _ = stats.spearmanr(X[:, i], y)
        correlations.append(abs(corr))
    
    # Sort by absolute correlation, highest first
    indices = np.argsort(correlations)[::-1]
    
    selected_indices = indices[:n_features]
    selected_features = [feature_names[i] for i in selected_indices]
    return selected_features, selected_indices

# ----- STEP 1: Select Top 5 Features Using 5 Methods -----
print("\n----- STEP 1: Selecting Top 5 Features Using 5 Different Methods -----")

# Now get both the features and their indices
rf_features_5, rf_indices_5 = select_features_rf(X_processed, y, feature_names, 5)
log_features_5, log_indices_5 = select_features_logistic(X_processed, y, feature_names, 5)
agg_features_5, agg_indices_5 = select_features_agglomeration(X_processed, y, feature_names, 5)
var_features_5, var_indices_5 = select_features_variance(X_processed, y, feature_names, 5)
spearman_features_5, spearman_indices_5 = select_features_spearman(X_processed, y, feature_names, 5)

print("\nTop 5 features from each method:")
print(f"Random Forest: {rf_features_5}")
print(f"Logistic Regression: {log_features_5}")
print(f"Feature Agglomeration: {agg_features_5}")
print(f"Variance: {var_features_5}")
print(f"Spearman Correlation: {spearman_features_5}")

# Function to extract selected features
def extract_features(X, feature_indices):
    X_selected = X[:, feature_indices]
    return X_selected

# ----- STEP 2: Cross-Validation with Top 5 Features -----
print("\n----- STEP 2: Cross-Validation with Top 5 Features -----")

# Random Forest Cross-Validation
def rf_cross_val(X_features, y):
    clf = RandomForestClassifier(random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_features, y, cv=cv, scoring='accuracy')
    return np.mean(cv_scores)

# Logistic Regression Cross-Validation
def log_cross_val(X_features, y):
    clf = LogisticRegression(random_state=42, max_iter=1000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_features, y, cv=cv, scoring='accuracy')
    return np.mean(cv_scores)

# Extract top 5 features for each method using indices
X_rf_5 = extract_features(X_processed, rf_indices_5)
X_log_5 = extract_features(X_processed, log_indices_5)
X_agg_5 = extract_features(X_processed, agg_indices_5)
X_var_5 = extract_features(X_processed, var_indices_5)
X_spearman_5 = extract_features(X_processed, spearman_indices_5)

# Perform cross-validation based on method
cv_rf_5 = rf_cross_val(X_rf_5, y)
cv_log_5 = log_cross_val(X_log_5, y)
cv_agg_5 = rf_cross_val(X_agg_5, y)
cv_var_5 = rf_cross_val(X_var_5, y)
cv_spearman_5 = rf_cross_val(X_spearman_5, y)

print(f"Random Forest features - RF CV: {cv_rf_5:.4f}")
print(f"Logistic Regression features - Logistic CV: {cv_log_5:.4f}")
print(f"Feature Agglomeration features - RF CV: {cv_agg_5:.4f}")
print(f"Variance features - RF CV: {cv_var_5:.4f}")
print(f"Spearman features - RF CV: {cv_spearman_5:.4f}")

# ----- STEP 3: Create Reduced Datasets by Removing Top 2 Features -----
print("\n----- STEP 3: Creating Reduced Datasets by Removing Top 2 Features -----")

# Get the top 2 feature indices for each method
rf_top2_indices = rf_indices_5[:2]
log_top2_indices = log_indices_5[:2]
agg_top2_indices = agg_indices_5[:2]
var_top2_indices = var_indices_5[:2]
spearman_top2_indices = spearman_indices_5[:2]

# Get the top 2 feature names for each method
rf_top2_features = rf_features_5[:2]
log_top2_features = log_features_5[:2]
agg_top2_features = agg_features_5[:2]
var_top2_features = var_features_5[:2]
spearman_top2_features = spearman_features_5[:2]

# Create reduced datasets by removing top 2 features
# We need to be careful about the indices shifting after removals
# Sort indices in descending order to avoid issues with shifting indices
rf_top2_indices_sorted = sorted(rf_top2_indices, reverse=True)
log_top2_indices_sorted = sorted(log_top2_indices, reverse=True)
agg_top2_indices_sorted = sorted(agg_top2_indices, reverse=True)
var_top2_indices_sorted = sorted(var_top2_indices, reverse=True)
spearman_top2_indices_sorted = sorted(spearman_top2_indices, reverse=True)

# Create reduced datasets by removing top 2 features
X_reduced_rf = np.delete(X_processed, rf_top2_indices_sorted, axis=1)
X_reduced_log = np.delete(X_processed, log_top2_indices_sorted, axis=1)
X_reduced_agg = np.delete(X_processed, agg_top2_indices_sorted, axis=1)
X_reduced_var = np.delete(X_processed, var_top2_indices_sorted, axis=1)
X_reduced_spearman = np.delete(X_processed, spearman_top2_indices_sorted, axis=1)

# Create new feature names for reduced datasets (excluding the removed features)
feature_names_rf = [f for i, f in enumerate(feature_names) if i not in rf_top2_indices]
feature_names_log = [f for i, f in enumerate(feature_names) if i not in log_top2_indices]
feature_names_agg = [f for i, f in enumerate(feature_names) if i not in agg_top2_indices]
feature_names_var = [f for i, f in enumerate(feature_names) if i not in var_top2_indices]
feature_names_spearman = [f for i, f in enumerate(feature_names) if i not in spearman_top2_indices]

print(f"Removed top 2 features from RF: {rf_top2_features}")
print(f"Removed top 2 features from Logistic: {log_top2_features}")
print(f"Removed top 2 features from Agglomeration: {agg_top2_features}")
print(f"Removed top 2 features from Variance: {var_top2_features}")
print(f"Removed top 2 features from Spearman: {spearman_top2_features}")


# ----- STEP 4: Select Top 3 Features from Reduced Datasets -----
print("\n----- STEP 4: Selecting Top 3 Features from Each Reduced Dataset -----")

# Since we removed 2 features, now we select top 3 features from each reduced dataset
rf_features_3, rf_indices_3 = select_features_rf(X_reduced_rf, y, feature_names_rf, 3)
log_features_3, log_indices_3 = select_features_logistic(X_reduced_log, y, feature_names_log, 3)
agg_features_3, agg_indices_3 = select_features_agglomeration(X_reduced_agg, y, feature_names_agg, 3)
var_features_3, var_indices_3 = select_features_variance(X_reduced_var, y, feature_names_var, 3)
spearman_features_3, spearman_indices_3 = select_features_spearman(X_reduced_spearman, y, feature_names_spearman, 3)

# Print selected features from each method on each reduced dataset
print(f"RF top 3 from RF reduced: {rf_features_3}")
print(f"LOG top 3 from LOG reduced: {log_features_3}")
print(f"AGG top 3 from AGG reduced: {agg_features_3}")
print(f"VAR top 3 from VAR reduced: {var_features_3}")
print(f"SPEARMAN top 3 from SPEARMAN reduced: {spearman_features_3}")

# Double-check that top 2 features have been removed
print("\nVerifying top 2 features are not in reduced datasets:")
for i, top_feature in enumerate(rf_top2_features):
    print(f"RF top feature {i+1} '{top_feature}' in RF top 3? {'Yes' if top_feature in rf_features_3 else 'No'}")

for i, top_feature in enumerate(log_top2_features):
    print(f"LOG top feature {i+1} '{top_feature}' in LOG top 3? {'Yes' if top_feature in log_features_3 else 'No'}")

for i, top_feature in enumerate(agg_top2_features):
    print(f"AGG top feature {i+1} '{top_feature}' in AGG top 3? {'Yes' if top_feature in agg_features_3 else 'No'}")

for i, top_feature in enumerate(var_top2_features):
    print(f"VAR top feature {i+1} '{top_feature}' in VAR top 3? {'Yes' if top_feature in var_features_3 else 'No'}")

for i, top_feature in enumerate(spearman_top2_features):
    print(f"SPEARMAN top feature {i+1} '{top_feature}' in SPEARMAN top 3? {'Yes' if top_feature in spearman_features_3 else 'No'}")

# ----- STEP 5: Extract Top 3 Features and Cross-Validate -----
print("\n----- STEP 5: Cross-Validation with Top 3 Features -----")

# Extract top 3 features from reduced datasets using indices
X_rf_3 = extract_features(X_reduced_rf, rf_indices_3)
X_log_3 = extract_features(X_reduced_log, log_indices_3)
X_agg_3 = extract_features(X_reduced_agg, agg_indices_3)
X_var_3 = extract_features(X_reduced_var, var_indices_3)
X_spearman_3 = extract_features(X_reduced_spearman, spearman_indices_3)

# Cross-validation for top 3 features
cv_rf_3 = rf_cross_val(X_rf_3, y)
cv_log_3 = log_cross_val(X_log_3, y)
cv_agg_3 = rf_cross_val(X_agg_3, y)
cv_var_3 = rf_cross_val(X_var_3, y)
cv_spearman_3 = rf_cross_val(X_spearman_3, y)

print(f"RF top 3 - RF CV: {cv_rf_3:.4f}")
print(f"LOG top 3 - LOG CV: {cv_log_3:.4f}")
print(f"AGG top 3 - RF CV: {cv_agg_3:.4f}")
print(f"VAR top 3 - RF CV: {cv_var_3:.4f}")
print(f"SPEARMAN top 3 - RF CV: {cv_spearman_3:.4f}")

# ----- STEP 6: Feature Ranking Stability Analysis -----
print("\n----- STEP 6: Feature Ranking Stability Analysis -----")

# Check which features remain in the top k after removing the top 2 features
def check_stability(original_top5, removed_top2_features, reduced_top3):
    # Get the features that should be common (original top 5 excluding the removed top 2 features)
    original_remaining = [f for f in original_top5 if f not in removed_top2_features]
    
    # Count how many features from the original remaining are still in the reduced top 3
    common_features = set(original_remaining) & set(reduced_top3)
    
    stability_score = len(common_features) / 3  # Normalize by max possible (3)
    
    return {
        'common_features': common_features,
        'stability_score': stability_score,
        'original_remaining': original_remaining,
        'reduced_top3': reduced_top3
    }

# Analyze stability for each method
rf_stability = check_stability(rf_features_5, rf_top2_features, rf_features_3)
log_stability = check_stability(log_features_5, log_top2_features, log_features_3)
agg_stability = check_stability(agg_features_5, agg_top2_features, agg_features_3)
var_stability = check_stability(var_features_5, var_top2_features, var_features_3)
spearman_stability = check_stability(spearman_features_5, spearman_top2_features, spearman_features_3)

# Print stability results
print("\nFeature Ranking Stability Results:")
print(f"Random Forest stability: {rf_stability['stability_score']:.2f}")
print(f"  Common features: {len(rf_stability['common_features'])}/3")
print(f"  Original top 5 features: {rf_features_5}")
print(f"  Removed top 2 features: {rf_top2_features}")
print(f"  Original remaining: {rf_stability['original_remaining']}")
print(f"  New top 3: {rf_stability['reduced_top3']}")

print(f"\nLogistic Regression stability: {log_stability['stability_score']:.2f}")
print(f"  Common features: {len(log_stability['common_features'])}/3")
print(f"  Original top 5 features: {log_features_5}")
print(f"  Removed top 2 features: {log_top2_features}")
print(f"  Original remaining: {log_stability['original_remaining']}")
print(f"  New top 3: {log_stability['reduced_top3']}")

print(f"\nFeature Agglomeration stability: {agg_stability['stability_score']:.2f}")
print(f"  Common features: {len(agg_stability['common_features'])}/3")
print(f"  Original top 5 features: {agg_features_5}")
print(f"  Removed top 2 features: {agg_top2_features}")
print(f"  Original remaining: {agg_stability['original_remaining']}")
print(f"  New top 3: {agg_stability['reduced_top3']}")

print(f"\nVariance stability: {var_stability['stability_score']:.2f}")
print(f"  Common features: {len(var_stability['common_features'])}/3")
print(f"  Original top 5 features: {var_features_5}")
print(f"  Removed top 2 features: {var_top2_features}")
print(f"  Original remaining: {var_stability['original_remaining']}")
print(f"  New top 3: {var_stability['reduced_top3']}")

print(f"\nSpearman stability: {spearman_stability['stability_score']:.2f}")
print(f"  Common features: {len(spearman_stability['common_features'])}/3")
print(f"  Original top 5 features: {spearman_features_5}")
print(f"  Removed top 2 features: {spearman_top2_features}")
print(f"  Original remaining: {spearman_stability['original_remaining']}")
print(f"  New top 3: {spearman_stability['reduced_top3']}")

# Calculate average stability across methods
avg_stability = (rf_stability['stability_score'] + log_stability['stability_score'] + 
                 agg_stability['stability_score'] + var_stability['stability_score'] + 
                 spearman_stability['stability_score']) / 5

print(f"\nAverage feature ranking stability across all methods: {avg_stability:.2f}")

# ----- STEP 7: Final Summary -----
print("\n----- STEP 7: Final Summary -----")

# Cross-validation results summary
print("\nCross-validation Results Summary:")
print("Top 5 Features:")
print(f"RF: {cv_rf_5:.4f}")
print(f"LOG: {cv_log_5:.4f}")
print(f"AGG: {cv_agg_5:.4f}")
print(f"VAR: {cv_var_5:.4f}")
print(f"SPEARMAN: {cv_spearman_5:.4f}")

print("\nTop 3 Features (after removing top 2 features):")
print(f"RF: {cv_rf_3:.4f}")
print(f"LOG: {cv_log_3:.4f}")
print(f"AGG: {cv_agg_3:.4f}")
print(f"VAR: {cv_var_3:.4f}")
print(f"SPEARMAN: {cv_spearman_3:.4f}")

