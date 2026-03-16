import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import gc
import warnings
warnings.filterwarnings('ignore')

# ========================
# 1. 加载数据
# ========================
print("Loading data...")
train = pd.read_csv('../resources/train_preprocessed.csv')
test = pd.read_csv('../resources/test_preprocessed.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# ========================
# 2. 准备特征和标签
# ========================
y = train['isFraud']
train_ids = train['TransactionID']
test_ids = test['TransactionID']

train = train.drop(['TransactionID', 'isFraud'], axis=1)
test = test.drop(['TransactionID'], axis=1)

common_cols = [c for c in train.columns if c in test.columns]
train = train[common_cols]
test = test[common_cols]

print(f"Features: {len(common_cols)}")

X = train.values
X_test = test.values
y = y.values

del train, test
gc.collect()

# ========================
# 3. LightGBM参数
# ========================
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.02,
    'num_leaves': 256,
    'max_depth': -1,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 2000,
    'verbose': -1,
    'random_state': 42,
    'n_jobs': -1,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5
}

# ========================
# 4. 交叉验证训练
# ========================
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_preds = np.zeros(len(y))
test_preds = np.zeros(len(X_test))
fold_scores = []

print(f"\nTraining with {n_folds}-fold CV...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold + 1}/{n_folds} ---")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model = lgb.LGBMClassifier(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(100)
        ]
    )
    
    val_pred = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_pred
    
    test_preds += model.predict_proba(X_test)[:, 1] / n_folds
    
    fold_auc = roc_auc_score(y_val, val_pred)
    fold_scores.append(fold_auc)
    print(f"Fold {fold + 1} AUC: {fold_auc:.5f}")
    
    del X_train, X_val, y_train, y_val, model
    gc.collect()

# ========================
# 5. 输出结果
# ========================
overall_auc = roc_auc_score(y, oof_preds)
print(f"\n{'='*50}")
print(f"Overall OOF AUC: {overall_auc:.5f}")
print(f"Mean Fold AUC: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")

# ========================
# 6. 保存预测结果
# ========================
submission = pd.DataFrame({
    'TransactionID': test_ids,
    'isFraud': test_preds
})
submission.to_csv('../resources/submission_v2.0.csv', index=False)
print(f"\nSubmission saved! Shape: {submission.shape}")
print(submission.head())
