import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_selection import SelectKBest
from scipy.stats import spearmanr, kendalltau
from sklearn.preprocessing import LabelEncoder

# Load the dataset with latin1 encoding
df = pd.read_csv("Liver Patient Dataset (LPD)_train.csv", encoding='latin1')

# Convert categorical variables to numerical
le = LabelEncoder()
df['Gender of the patient'] = le.fit_transform(df['Gender of the patient'])

# Separate features and target
X = df.drop('Result', axis=1)
y = df['Result']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
imputer.fit(X)
X_imputed = imputer.transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

# Function to evaluate features using Random Forest
def evaluate_rf_features(X, y, n_features=5):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    feature_importances = rf.feature_importances_
    indices = np.argsort(feature_importances)[::-1]
    
    top_features = [(X.columns[i], feature_importances[i]) for i in indices[:n_features]]
    
    return top_features

# Function to evaluate features using Feature Agglomeration
def evaluate_agglom_features(X, y, n_features=5):
    # Get correlation between features
    corr_matrix = X.corr().abs()
    
    # Use Feature Agglomeration to group similar features
    agglo = FeatureAgglomeration(n_clusters=n_features)
    agglo.fit(X)
    
    # For each cluster, find the most representative feature
    cluster_to_feature = {}
    for feature_idx, cluster_idx in enumerate(agglo.labels_):
        if cluster_idx in cluster_to_feature:
            current_feature = X.columns[feature_idx]
            existing_feature = cluster_to_feature[cluster_idx]
            # Use the feature with higher average correlation as representative
            if corr_matrix[current_feature].mean() > corr_matrix[existing_feature].mean():
                cluster_to_feature[cluster_idx] = current_feature
        else:
            cluster_to_feature[cluster_idx] = X.columns[feature_idx]
    
    # Get the top features (representatives of each cluster)
    top_features = [(feature, corr_matrix[feature].mean()) for cluster, feature in cluster_to_feature.items()]
    top_features.sort(key=lambda x: x[1], reverse=True)
    
    return top_features

# Function to evaluate features based on variance (similar to highly variable gene selection)
def evaluate_variance_features(X, y, n_features=5):
    # Calculate variance of each feature
    variances = X.var()
    indices = np.argsort(variances.values)[::-1]
    
    top_features = [(X.columns[i], variances[X.columns[i]]) for i in indices[:n_features]]
    
    return top_features

# Function to evaluate features using Spearman correlation with p-values
def evaluate_spearman_features(X, y, n_features=5):
    feature_scores = []
    
    for col in X.columns:
        corr, p_value = spearmanr(X[col], y)
        feature_scores.append((col, abs(corr), p_value))
    
    # Sort by correlation (absolute value), then by p-value (ascending)
    feature_scores.sort(key=lambda x: (x[1], -x[2]), reverse=True)
    
    top_features = [(feature, score) for feature, score, _ in feature_scores[:n_features]]
    
    return top_features

# Function to evaluate features using Logistic Regression
def evaluate_logistic_features(X, y, n_features=5):
    # Train logistic regression with L1 penalty
    model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=42)
    model.fit(X, y)
    
    # Get feature importance from coefficients
    feature_importances = np.abs(model.coef_[0])
    indices = np.argsort(feature_importances)[::-1]
    
    top_features = [(X.columns[i], feature_importances[i]) for i in indices[:n_features]]
    
    return top_features

# Function to evaluate model performance with cross-validation
def evaluate_model_with_features(X, y, feature_names, model_type="rf"):
    X_selected = X[feature_names]
    
    if model_type == "rf":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "lr":
        model = LogisticRegression(random_state=42)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_selected, y, cv=cv, scoring='accuracy')
    
    return scores.mean(), scores.std()

# Function to compare ranking stability using Kendall's tau
def calculate_ranking_stability(list1, list2):
    # Create mapping of features to their ranks
    rank_map1 = {item: idx for idx, item in enumerate(list1)}
    rank_map2 = {item: idx for idx, item in enumerate(list2)}
    
    # Get common features
    common_features = set(list1).intersection(set(list2))
    
    if len(common_features) < 2:
        return 0.0  # Not enough common features for meaningful comparison
    
    # Extract ranks of common features
    ranks1 = [rank_map1[item] for item in common_features]
    ranks2 = [rank_map2[item] for item in common_features]
    
    # Calculate Kendall's tau correlation coefficient
    tau, _ = kendalltau(ranks1, ranks2)
    
    # Handle NaN (can occur if all ranks are the same)
    if np.isnan(tau):
        return 1.0 if ranks1 == ranks2 else 0.0
    
    # Convert from -1...1 to 0...1 scale for easier interpretation
    stability = (tau + 1) / 2
    
    return stability

# Perform feature selection with all methods
print("Feature Selection Results:")

methods = ["Random Forest", "Feature Agglomeration", "Variance-Based", "Spearman Correlation", "Logistic Regression"]
top5_features_by_method = []
highest_ranked_features = []

# 1. Random Forest
rf_features = evaluate_rf_features(X, y, 5)
top5_features_by_method.append([feature for feature, _ in rf_features])
highest_ranked_features.append(rf_features[0][0])
rf_acc_mean, rf_acc_std = evaluate_model_with_features(X, y, [feature for feature, _ in rf_features], "rf")
print("\n1. Random Forest - Top 5 Features:")
for i, (feature, importance) in enumerate(rf_features):
    print(f"   {i+1}. {feature}: {importance:.4f}")
print(f"   CV Accuracy with top 5: {rf_acc_mean:.4f} ± {rf_acc_std:.4f}")

# Also evaluate top 4 features
rf_top4_features = [feature for feature, _ in rf_features[:4]]
rf_top4_acc_mean, rf_top4_acc_std = evaluate_model_with_features(X, y, rf_top4_features, "rf")
print(f"   CV Accuracy with top 4: {rf_top4_acc_mean:.4f} ± {rf_top4_acc_std:.4f}")

# 2. Feature Agglomeration
agglom_features = evaluate_agglom_features(X, y, 5)
top5_features_by_method.append([feature for feature, _ in agglom_features])
highest_ranked_features.append(agglom_features[0][0])
agglom_acc_mean, agglom_acc_std = evaluate_model_with_features(X, y, [feature for feature, _ in agglom_features], "rf")
print("\n2. Feature Agglomeration - Top 5 Features:")
for i, (feature, importance) in enumerate(agglom_features):
    print(f"   {i+1}. {feature}: {importance:.4f}")
print(f"   CV Accuracy with top 5: {agglom_acc_mean:.4f} ± {agglom_acc_std:.4f}")

# Also evaluate top 4 features
agglom_top4_features = [feature for feature, _ in agglom_features[:4]]
agglom_top4_acc_mean, agglom_top4_acc_std = evaluate_model_with_features(X, y, agglom_top4_features, "rf")
print(f"   CV Accuracy with top 4: {agglom_top4_acc_mean:.4f} ± {agglom_top4_acc_std:.4f}")

# 3. Variance-Based (similar to highly variable gene selection)
var_features = evaluate_variance_features(X, y, 5)
top5_features_by_method.append([feature for feature, _ in var_features])
highest_ranked_features.append(var_features[0][0])
var_acc_mean, var_acc_std = evaluate_model_with_features(X, y, [feature for feature, _ in var_features], "rf")
print("\n3. Variance-Based - Top 5 Features:")
for i, (feature, importance) in enumerate(var_features):
    print(f"   {i+1}. {feature}: {importance:.4f}")
print(f"   CV Accuracy with top 5: {var_acc_mean:.4f} ± {var_acc_std:.4f}")

# Also evaluate top 4 features
var_top4_features = [feature for feature, _ in var_features[:4]]
var_top4_acc_mean, var_top4_acc_std = evaluate_model_with_features(X, y, var_top4_features, "rf")
print(f"   CV Accuracy with top 4: {var_top4_acc_mean:.4f} ± {var_top4_acc_std:.4f}")

# 4. Spearman Correlation
spearman_features = evaluate_spearman_features(X, y, 5)
top5_features_by_method.append([feature for feature, _ in spearman_features])
highest_ranked_features.append(spearman_features[0][0])
spearman_acc_mean, spearman_acc_std = evaluate_model_with_features(X, y, [feature for feature, _ in spearman_features], "rf")
print("\n4. Spearman Correlation - Top 5 Features:")
for i, (feature, importance) in enumerate(spearman_features):
    print(f"   {i+1}. {feature}: {importance:.4f}")
print(f"   CV Accuracy with top 5: {spearman_acc_mean:.4f} ± {spearman_acc_std:.4f}")

# Also evaluate top 4 features
spearman_top4_features = [feature for feature, _ in spearman_features[:4]]
spearman_top4_acc_mean, spearman_top4_acc_std = evaluate_model_with_features(X, y, spearman_top4_features, "rf")
print(f"   CV Accuracy with top 4: {spearman_top4_acc_mean:.4f} ± {spearman_top4_acc_std:.4f}")

# 5. Logistic Regression
lr_features = evaluate_logistic_features(X, y, 5)
top5_features_by_method.append([feature for feature, _ in lr_features])
highest_ranked_features.append(lr_features[0][0])
lr_acc_mean, lr_acc_std = evaluate_model_with_features(X, y, [feature for feature, _ in lr_features], "lr")
print("\n5. Logistic Regression - Top 5 Features:")
for i, (feature, importance) in enumerate(lr_features):
    print(f"   {i+1}. {feature}: {importance:.4f}")
print(f"   CV Accuracy with top 5: {lr_acc_mean:.4f} ± {lr_acc_std:.4f}")

# Also evaluate top 4 features
lr_top4_features = [feature for feature, _ in lr_features[:4]]
lr_top4_acc_mean, lr_top4_acc_std = evaluate_model_with_features(X, y, lr_top4_features, "lr")
print(f"   CV Accuracy with top 4: {lr_top4_acc_mean:.4f} ± {lr_top4_acc_std:.4f}")

# Remove the highest ranked feature for each algorithm to create reduced datasets
print("\n\nStability Analysis - Removing Highest Feature For Each Method:")

for i, method in enumerate(methods):
    print(f"\n{method} - Removing '{highest_ranked_features[i]}':")
    
    # Create reduced dataset without the highest ranked feature
    X_reduced = X.drop(highest_ranked_features[i], axis=1)
    
    # Select top 4 features from reduced dataset based on the method
    if method == "Random Forest":
        reduced_features = evaluate_rf_features(X_reduced, y, 4)
    elif method == "Feature Agglomeration":
        reduced_features = evaluate_agglom_features(X_reduced, y, 4)
    elif method == "Variance-Based":
        reduced_features = evaluate_variance_features(X_reduced, y, 4)
    elif method == "Spearman Correlation":
        reduced_features = evaluate_spearman_features(X_reduced, y, 4)
    else:  # Logistic Regression
        reduced_features = evaluate_logistic_features(X_reduced, y, 4)
    
    # Compare with original top 5 (excluding the highest)
    original_top5_minus_highest = [f for f in top5_features_by_method[i][1:5]]
    new_top4 = [feature for feature, _ in reduced_features]
    
    print(f"   Original top 5 (excluding highest): {original_top5_minus_highest}")
    print(f"   New top 4 from reduced dataset: {new_top4}")
    
    # Calculate rank stability using Kendall's tau
    rank_stability = calculate_ranking_stability(original_top5_minus_highest, new_top4)
    print(f"   Rank stability score (Kendall's tau): {rank_stability:.4f}")
    
    # Count common features for additional info
    common_features = set(original_top5_minus_highest).intersection(new_top4)
    overlap_ratio = len(common_features) / 4.0
    print(f"   Feature overlap: {overlap_ratio:.2f} ({len(common_features)}/4 features in common)")
    
    # Evaluate performance with new top 4 features
    model_type = "lr" if method == "Logistic Regression" else "rf"
    new_acc_mean, new_acc_std = evaluate_model_with_features(X_reduced, y, new_top4, model_type)
    print(f"   CV Accuracy with new top 4: {new_acc_mean:.4f} ± {new_acc_std:.4f}")
