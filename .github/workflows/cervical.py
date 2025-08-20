import openml
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import StratifiedKFold, cross_val_score

# —— Feature‐selection routines —— 

def get_top_features_rf(X, y, k, rs=42):
    rf = RandomForestClassifier(n_estimators=200, random_state=rs)
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns)
    return imp.nlargest(k).index.tolist()

def get_top_features_lr(X, y, k, rs=42):
    lr = LogisticRegression(max_iter=2000, random_state=rs)
    lr.fit(X, y)
    coefs = pd.Series(np.abs(lr.coef_[0]), index=X.columns)
    return coefs.nlargest(k).index.tolist()

def get_top_features_spearman(X, y, k):
    pvals = {f: spearmanr(X[f], y)[1] for f in X.columns}
    return pd.Series(pvals).nsmallest(k).index.tolist()

def get_top_features_hvg(X, k):
    return X.var(axis=0).nlargest(k).index.tolist()

def get_top_features_agglom(X, k, rs=42):
    n_feats = X.shape[1]
    n_clusters = min(n_feats, 2*k)
    corr = X.corr().abs()
    cl = AgglomerativeClustering(n_clusters=n_clusters)
    cl.fit(X.T)
    labels = cl.labels_
    centrality = {}
    for cid in np.unique(labels):
        members = X.columns[labels == cid]
        if len(members) == 1:
            centrality[members[0]] = 0.0
        else:
            sub = corr.loc[members, members]
            sums = sub.sum(axis=1)
            for f, s in sums.items():
                centrality[f] = s
    return pd.Series(centrality).nlargest(k).index.tolist()

# —— Cross‐validation helper —— 

def cv_accuracy(clf, X, y, cv):
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    return scores.mean(), scores.std()

# —— Main pipeline —— 

def main():
    # 1) Download, impute, save full
    print("Downloading dataset 42912 …")
    ds = openml.datasets.get_dataset(42912)
    X, y, _, _ = ds.get_data(target='Biopsy', dataset_format='dataframe')
    imp = SimpleImputer(strategy='mean')
    imp.fit(X)
    X_imp = pd.DataFrame(imp.transform(X), columns=X.columns)
    pd.concat([X_imp, y.rename('Biopsy')], axis=1)\
      .to_csv('behavior_cervical_full.csv', index=False)

    # 2) CV splitter
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 3) Define selectors (func, is_supervised)
    selectors = {
        'RandomForest'       : (get_top_features_rf,       True),
        'LogisticRegression' : (get_top_features_lr,       True),
        'Spearman'           : (get_top_features_spearman, True),
        'HighVar'            : (get_top_features_hvg,      False),
        'Agglomeration'      : (get_top_features_agglom,   False)
    }

    summary = []
    for name, (fn, supervised) in selectors.items():
        print(f"\n=== {name} ===")

        # a) Select top 5 from full X
        top5 = fn(X_imp, y, 5) if supervised else fn(X_imp, 5)
        X5   = X_imp[top5]
        print(" top5:", top5)

        clf5 = (LogisticRegression(max_iter=2000, random_state=42)
                if name=='LogisticRegression'
                else RandomForestClassifier(n_estimators=200, random_state=42))
        m5, s5 = cv_accuracy(clf5, X5, y, cv)
        print(f" CV@5 acc = {m5:.4f} ± {s5:.4f}")

        # b) Drop the top-1 feature, form reduced X
        drop1 = top5[0:1]
        Xr    = X_imp.drop(columns=drop1)

        # c) Select top 4 from reduced X
        top4  = fn(Xr, y, 4) if supervised else fn(Xr, 4)
        X4    = Xr[top4]
        print(" top4 (after drop1):", top4)

        clf4 = (LogisticRegression(max_iter=2000, random_state=42)
                if name=='LogisticRegression'
                else RandomForestClassifier(n_estimators=200, random_state=42))
        m4, s4 = cv_accuracy(clf4, X4, y, cv)
        print(f" CV@4 acc = {m4:.4f} ± {s4:.4f}")

        # d) New stability: compare remaining 4 positions
        rem5 = top5[1:]  # ordered list of length 4
        matches = sum(1 for i in range(4) if rem5[i] == top4[i])
        stab = matches / 4.0
        print(f" Stability = {matches}/4 = {stab:.3f}")

        summary.append({
            'Method'    : name,
            'CV5_mean'  : m5,   'CV5_std'  : s5,
            'CV4_mean'  : m4,   'CV4_std'  : s4,
            'Stability' : stab
        })

    # 4) Summary
    df = pd.DataFrame(summary).set_index('Method')
    print("\n=== Final summary ===")
    print(df)


if __name__ == "__main__":
    main()
