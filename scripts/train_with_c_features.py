import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("C特征模型训练 - 对比多种模型")
print("=" * 60)

# ========================
# 1. 加载数据
# ========================
print("\n[1] Loading data...")
train = pd.read_csv('../resources/train_preprocessed.csv')
test = pd.read_csv('../resources/test_preprocessed.csv')

print(f"Train: {train.shape}, Test: {test.shape}")

# 保存标签和ID
y = train['isFraud'].values
test_ids = test['TransactionID'].values

train = train.drop(['Unnamed: 0', 'TransactionID', 'isFraud'], axis=1)
test = test.drop(['Unnamed: 0', 'TransactionID'], axis=1)

common_cols = [c for c in train.columns if c in test.columns]
train = train[common_cols]
test = test[common_cols]

# ========================
# 2. 提取C特征并添加PCA
# ========================
print("\n[2] Extracting C features...")

c_cols = [c for c in train.columns if c.startswith('C')]
print(f"C features: {len(c_cols)}")

# 标准化
scaler = StandardScaler()
train_c = pd.DataFrame(
    scaler.fit_transform(train[c_cols]),
    columns=c_cols
)
test_c = pd.DataFrame(
    scaler.transform(test[c_cols]),
    columns=c_cols
)

# PCA on C features
n_pca = 10
pca = PCA(n_components=n_pca, random_state=42)
train_c_pca = pca.fit_transform(train_c)
test_c_pca = pca.transform(test_c)

print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")

# 创建PCA特征名
pca_cols = [f'C_pca_{i}' for i in range(n_pca)]
train_c_pca_df = pd.DataFrame(train_c_pca, columns=pca_cols)
test_c_pca_df = pd.DataFrame(test_c_pca, columns=pca_cols)

# 合并原始C特征 + PCA特征
train_c_all = pd.concat([train_c, train_c_pca_df], axis=1)
test_c_all = pd.concat([test_c, test_c_pca_df], axis=1)

print(f"Total C features: {train_c_all.shape[1]}")

# ========================
# 3. 添加一些关键特征
# ========================
print("\n[3] Adding key features...")

# 时间特征
train['hour'] = (train['TransactionDT'] // 3600) % 24
test['hour'] = (test['TransactionDT'] // 3600) % 24

# 金额特征
train['TransactionAmt_log'] = np.log1p(train['TransactionAmt'])
test['TransactionAmt_log'] = np.log1p(test['TransactionAmt'])

# 合并所有特征
X_train = pd.concat([train_c_all, train[['hour', 'TransactionAmt', 'TransactionAmt_log']]], axis=1)
X_test = pd.concat([test_c_all, test[['hour', 'TransactionAmt', 'TransactionAmt_log']]], axis=1)

# 填充NaN
X_train = X_train.fillna(-999)
X_test = X_test.fillna(-999)

print(f"Final features: {X_train.shape[1]}")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

X_train = X_train.values
X_test = X_test.values

# ========================
# 4. 模型对比
# ========================
print("\n[4] Training models with 5-fold CV...")

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# 存储各模型结果
models = {
    'LogisticRegression': LogisticRegression(max_iter=500, random_state=42, n_jobs=-1),
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=64,
        max_depth=8,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
}

results = {}

for model_name, model_template in models.items():
    print(f"\n--- {model_name} ---")
    
    oof_preds = np.zeros(len(y))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        # 克隆模型
        if model_name == 'LogisticRegression':
            model = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1)
        elif model_name == 'RandomForest':
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        else:
            model = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=64,
                max_depth=8,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
        
        model.fit(X_tr, y_tr)
        
        val_pred = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_pred
        test_preds += model.predict_proba(X_test)[:, 1] / n_folds
        
        fold_auc = roc_auc_score(y_val, val_pred)
        fold_scores.append(fold_auc)
        print(f"  Fold {fold+1}: AUC = {fold_auc:.5f}")
    
    overall_auc = roc_auc_score(y, oof_preds)
    results[model_name] = {
        'oof': oof_preds,
        'test': test_preds,
        'auc': overall_auc,
        'mean_auc': np.mean(fold_scores)
    }
    print(f"  Overall OOF AUC: {overall_auc:.5f}")

# ========================
# 5. 结果汇总
# ========================
print("\n" + "=" * 60)
print("模型对比结果 (C特征 + PCA)")
print("=" * 60)

for name, res in sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True):
    print(f"{name:20s}: OOF AUC = {res['auc']:.5f}")

# 保存最佳模型结果
best_model = max(results.items(), key=lambda x: x[1]['auc'])
print(f"\nBest model: {best_model[0]} with AUC = {best_model[1]['auc']:.5f}")

# 保存submission
submission = pd.DataFrame({
    'TransactionID': test_ids,
    'isFraud': best_model[1]['test']
})
submission.to_csv('../resources/submission_v3.0.csv', index=False)
print(f"\nSubmission saved to ../resources/submission_v3.0.csv")
print(submission.head())
