import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import gc
import sys
import warnings
warnings.filterwarnings('ignore')

# 日志输出到文件
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

log = Logger('../result/log_catboost.txt')
sys.stdout = log

print("=" * 60)
print("CatBoost模型训练 - C特征PCA降维 - 对比多种模型")
print("注意：不使用不平衡处理，加强正则化防止过拟合")
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

print(f"Total features: {len(common_cols)}")

# ========================
# 2. 特征工程
# ========================
print("\n[2] Feature Engineering...")

# 时间特征
train['hour'] = (train['TransactionDT'] // 3600) % 24
test['hour'] = (test['TransactionDT'] // 3600) % 24
train['day'] = (train['TransactionDT'] // 86400) % 7
test['day'] = (test['TransactionDT'] // 86400) % 7

# 金额特征
train['TransactionAmt_log'] = np.log1p(train['TransactionAmt'])
test['TransactionAmt_log'] = np.log1p(test['TransactionAmt'])
train['TransactionAmt_cents'] = train['TransactionAmt'] % 1
test['TransactionAmt_cents'] = test['TransactionAmt'] % 1

print(f"After feature engineering: {train.shape[1]} features")

# ========================
# 3. C特征PCA（删除原始C特征）
# ========================
print("\n[3] C features PCA (removing original C features)...")

c_cols = [c for c in train.columns if c.startswith('C')]
print(f"Original C features: {len(c_cols)}")

# 标准化C特征
scaler_c = StandardScaler()
train_c_scaled = scaler_c.fit_transform(train[c_cols])
test_c_scaled = scaler_c.transform(test[c_cols])

# PCA on C features
n_pca = 10
pca = PCA(n_components=n_pca, random_state=42)
train_c_pca = pca.fit_transform(train_c_scaled)
test_c_pca = pca.transform(test_c_scaled)

print(f"C-PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")

# 删除原始C特征
train = train.drop(columns=c_cols)
test = test.drop(columns=c_cols)
print(f"After removing C features: {train.shape[1]} features")

# 添加PCA特征
pca_cols = [f'C_pca_{i}' for i in range(n_pca)]
train_c_pca_df = pd.DataFrame(train_c_pca, columns=pca_cols)
test_c_pca_df = pd.DataFrame(test_c_pca, columns=pca_cols)

train = pd.concat([train.reset_index(drop=True), train_c_pca_df], axis=1)
test = pd.concat([test.reset_index(drop=True), test_c_pca_df], axis=1)

print(f"After adding C-PCA: {train.shape[1]} features")

# ========================
# 4. 准备数据
# ========================
print("\n[4] Preparing data...")

# 填充缺失值
train = train.fillna(-999)
test = test.fillna(-999)

# 获取所有特征列
feature_cols = train.columns.tolist()
print(f"Final features: {len(feature_cols)}")

# 转换回numpy
X_train = train.values.astype(np.float32)
X_test = test.values.astype(np.float32)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

del train, test
gc.collect()

# ========================
# 5. 模型对比
# ========================
print("\n[5] Training models with 5-fold CV...")

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

results = {}

# ========================
# Model 1: Logistic Regression
# ========================
print("\n--- Logistic Regression (scaled data) ---")

# 标准化用于LR
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

oof_lr = np.zeros(len(y))
test_lr = np.zeros(len(X_test))
fold_scores_lr = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y)):
    X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    model = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1, solver='lbfgs')
    model.fit(X_tr, y_tr)
    
    val_pred = model.predict_proba(X_val)[:, 1]
    oof_lr[val_idx] = val_pred
    test_lr += model.predict_proba(X_test_scaled)[:, 1] / n_folds
    
    fold_auc = roc_auc_score(y_val, val_pred)
    fold_scores_lr.append(fold_auc)
    print(f"  Fold {fold+1}: AUC = {fold_auc:.5f}")

auc_lr = roc_auc_score(y, oof_lr)
results['LogisticRegression'] = {'oof': oof_lr, 'test': test_lr, 'auc': auc_lr}
print(f"  Overall OOF AUC: {auc_lr:.5f}")

# ========================
# Model 2: Random Forest
# ========================
print("\n--- Random Forest ---")

oof_rf = np.zeros(len(y))
test_rf = np.zeros(len(X_test))
fold_scores_rf = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=50,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    
    val_pred = model.predict_proba(X_val)[:, 1]
    oof_rf[val_idx] = val_pred
    test_rf += model.predict_proba(X_test)[:, 1] / n_folds
    
    fold_auc = roc_auc_score(y_val, val_pred)
    fold_scores_rf.append(fold_auc)
    print(f"  Fold {fold+1}: AUC = {fold_auc:.5f}")

auc_rf = roc_auc_score(y, oof_rf)
results['RandomForest'] = {'oof': oof_rf, 'test': test_rf, 'auc': auc_rf}
print(f"  Overall OOF AUC: {auc_rf:.5f}")

# ========================
# Model 3: CatBoost
# ========================
print("\n--- CatBoost (with strong regularization) ---")

oof_cb = np.zeros(len(y))
test_cb = np.zeros(len(X_test))
fold_scores_cb = []

catboost_params = {
    'iterations': 1000,
    'learning_rate': 0.03,
    'depth': 6,
    'l2_leaf_reg': 10,           # L2正则化，增大防止过拟合
    'bagging_temperature': 0.8,  # Bagging温度
    'random_strength': 1.0,      # 随机强度
    'border_count': 128,
    'task_type': 'CPU',
    'random_seed': 42,
    'verbose': 200,
    'early_stopping_rounds': 100
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    print(f"\n  Fold {fold+1}/{n_folds}:")
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    model = CatBoostClassifier(**catboost_params)
    model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        use_best_model=True
    )
    
    val_pred = model.predict_proba(X_val)[:, 1]
    oof_cb[val_idx] = val_pred
    test_cb += model.predict_proba(X_test)[:, 1] / n_folds
    
    fold_auc = roc_auc_score(y_val, val_pred)
    fold_scores_cb.append(fold_auc)
    print(f"  Fold {fold+1} AUC: {fold_auc:.5f}")
    
    del model
    gc.collect()

auc_cb = roc_auc_score(y, oof_cb)
results['CatBoost'] = {'oof': oof_cb, 'test': test_cb, 'auc': auc_cb}
print(f"  Overall OOF AUC: {auc_cb:.5f}")

# ========================
# 6. 模型融合（简单平均）
# ========================
print("\n--- Model Ensemble (Simple Average) ---")

# CatBoost + RandomForest 融合
oof_ensemble = (oof_rf + oof_cb) / 2
test_ensemble = (test_rf + test_cb) / 2
auc_ensemble = roc_auc_score(y, oof_ensemble)
results['Ensemble'] = {'oof': oof_ensemble, 'test': test_ensemble, 'auc': auc_ensemble}
print(f"  Ensemble OOF AUC: {auc_ensemble:.5f}")

# ========================
# 7. 结果汇总
# ========================
print("\n" + "=" * 60)
print("模型对比结果")
print("=" * 60)

for name, res in sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True):
    print(f"{name:20s}: OOF AUC = {res['auc']:.5f}")

# 保存最佳模型结果
best_model = max(results.items(), key=lambda x: x[1]['auc'])
print(f"\nBest model: {best_model[0]} with AUC = {best_model[1]['auc']:.5f}")

# 保存submission到resources
best_submission = pd.DataFrame({
    'TransactionID': test_ids,
    'isFraud': best_model[1]['test']
})
best_submission.to_csv('../resources/submission_catboost.csv', index=False)

print(f"\nSubmission saved to ../resources/submission_catboost.csv")
print(best_submission.head())

log.log.close()