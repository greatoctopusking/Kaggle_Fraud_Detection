import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
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

log = Logger('../result/log.txt')
sys.stdout = log

print("=" * 60)
print("全部特征模型训练 - C特征PCA降维 - 对比多种模型")
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

# ========================
# 5. 数据归一化（用于逻辑回归和SVM）
# ========================
print("\n[5] Normalizing data for LR and SVM...")

# 分离数值列（用于标准化）
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train[numeric_cols])
test_scaled = scaler.transform(test[numeric_cols])

train_scaled_df = pd.DataFrame(train_scaled, columns=numeric_cols)
test_scaled_df = pd.DataFrame(test_scaled, columns=numeric_cols)

# 转换回numpy
X_train_all = train.values.astype(np.float32)
X_test_all = test.values.astype(np.float32)

X_train_scaled = train_scaled_df.values.astype(np.float32)
X_test_scaled = test_scaled_df.values.astype(np.float32)

print(f"X_train shape: {X_train_all.shape}")
print(f"X_test shape: {X_test_all.shape}")

del train, test, train_scaled_df, test_scaled_df
gc.collect()

# ========================
# 6. 模型对比
# ========================
print("\n[6] Training models with 5-fold CV...")

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

results = {}

# ========================
# Model 1: Logistic Regression
# ========================
print("\n--- Logistic Regression ---")

oof_lr = np.zeros(len(y))
test_lr = np.zeros(len(X_test_all))
fold_scores_lr = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y)):
    X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    model = LogisticRegression(
        max_iter=500, 
        random_state=42, 
        n_jobs=-1, 
        solver='lbfgs',
        class_weight='balanced'  # 不平衡处理
    )
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
test_rf = np.zeros(len(X_test_all))
fold_scores_rf = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_all, y)):
    X_tr, X_val = X_train_all[train_idx], X_train_all[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=50,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # 不平衡处理
    )
    model.fit(X_tr, y_tr)
    
    val_pred = model.predict_proba(X_val)[:, 1]
    oof_rf[val_idx] = val_pred
    test_rf += model.predict_proba(X_test_all)[:, 1] / n_folds
    
    fold_auc = roc_auc_score(y_val, val_pred)
    fold_scores_rf.append(fold_auc)
    print(f"  Fold {fold+1}: AUC = {fold_auc:.5f}")

auc_rf = roc_auc_score(y, oof_rf)
results['RandomForest'] = {'oof': oof_rf, 'test': test_rf, 'auc': auc_rf}
print(f"  Overall OOF AUC: {auc_rf:.5f}")

# ========================
# Model 3: LightGBM
# ========================
print("\n--- LightGBM ---")

oof_lgb = np.zeros(len(y))
test_lgb = np.zeros(len(X_test_all))
fold_scores_lgb = []

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.02,
    'num_leaves': 128,
    'max_depth': -1,
    'min_child_samples': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 10000,
    'verbose': -1,
    'random_state': 42,
    'n_jobs': -1,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'is_unbalance': True  # 不平衡处理
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_all, y)):
    X_tr, X_val = X_train_all[train_idx], X_train_all[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(200)
        ]
    )
    
    val_pred = model.predict_proba(X_val)[:, 1]
    oof_lgb[val_idx] = val_pred
    test_lgb += model.predict_proba(X_test_all)[:, 1] / n_folds
    
    fold_auc = roc_auc_score(y_val, val_pred)
    fold_scores_lgb.append(fold_auc)
    print(f"  Fold {fold+1}: AUC = {fold_auc:.5f}")

auc_lgb = roc_auc_score(y, oof_lgb)
results['LightGBM'] = {'oof': oof_lgb, 'test': test_lgb, 'auc': auc_lgb}
print(f"  Overall OOF AUC: {auc_lgb:.5f}")

# ========================
# 7. 结果汇总
# ========================
print("\n" + "=" * 60)
print("模型对比结果 (C特征PCA降维后)")
print("=" * 60)

for name, res in sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True):
    print(f"{name:20s}: OOF AUC = {res['auc']:.5f}")

# 保存最佳模型结果
best_model = max(results.items(), key=lambda x: x[1]['auc'])
print(f"\nBest model: {best_model[0]} with AUC = {best_model[1]['auc']:.5f}")

# 保存最佳submission到resources
best_submission = pd.DataFrame({
    'TransactionID': test_ids,
    'isFraud': best_model[1]['test']
})
best_submission.to_csv('../result/submission_v5.0.csv', index=False)

print(f"\nSubmission saved to ../result/submission_v5.0.csv")
print(f"Best model: {best_model[0]} with AUC = {best_model[1]['auc']:.5f}")
print(best_submission.head())

log.log.close()