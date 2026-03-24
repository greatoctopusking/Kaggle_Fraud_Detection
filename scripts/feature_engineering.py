import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
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

# 保存标签和ID
y = train['isFraud'].values
train_ids = train['TransactionID'].values
test_ids = test['TransactionID'].values

# 移除标签和ID，保留索引列
train = train.drop(['Unnamed: 0', 'TransactionID', 'isFraud'], axis=1)
test = test.drop(['Unnamed: 0', 'TransactionID'], axis=1)

# 确保列一致
common_cols = [c for c in train.columns if c in test.columns]
train = train[common_cols]
test = test[common_cols]

print(f"Features: {len(common_cols)}")

# ========================
# 2. 特征工程
# ========================
print("\n=== Feature Engineering ===")

def add_features(df):
    # 时间特征
    df['hour'] = (df['TransactionDT'] // 3600) % 24
    df['day'] = (df['TransactionDT'] // 86400) % 7
    df['day_of_month'] = (df['TransactionDT'] // 86400) % 30
    
    # 金额特征
    df['TransactionAmt_log'] = np.log1p(df['TransactionAmt'])
    df['TransactionAmt_cents'] = df['TransactionAmt'] % 1
    df['TransactionAmt_is_integer'] = (df['TransactionAmt'] == df['TransactionAmt'].astype(int)).astype(int)
    
    return df

# 合并以便统一处理
all_data = pd.concat([train, test], axis=0, ignore_index=True)
del train, test
import gc
gc.collect()

# 添加特征
all_data = add_features(all_data)
print(f"After feature engineering: {all_data.shape}")

# 分离回训练集和测试集
n_train = len(y)
train = all_data.iloc[:n_train].copy()
test = all_data.iloc[n_train:].copy()
del all_data
gc.collect()

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# ========================
# 3. 数据归一化
# ========================
print("\n=== Data Normalization ===")

# 识别数值列（排除标记为-999的缺失值处理后的列）
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numeric columns: {len(numeric_cols)}")

# 标准化
scaler = StandardScaler()
train_scaled = train.copy()
test_scaled = test.copy()

train_scaled[numeric_cols] = scaler.fit_transform(train[numeric_cols])
test_scaled[numeric_cols] = scaler.transform(test[numeric_cols])

print("Normalization completed!")

# ========================
# 4. PCA降维
# ========================
print("\n=== PCA Dimensionality Reduction ===")

# ========================
# 5. 多种特征的PCA和t-SNE对比
# ========================
print("\n=== Feature Comparison: V vs C+D vs All Numeric ===")

# 取样以加速
sample_size = 10000
np.random.seed(42)
sample_idx = np.random.choice(len(train), sample_size, replace=False)
y_sample = y[sample_idx]

# 定义多组特征
feature_sets = {
    'V_features': [c for c in train.columns if c.startswith('V')],
    'C_features': [c for c in train.columns if c.startswith('C')],
    'D_features': [c for c in train.columns if c.startswith('D')],
    'Time_Amt': ['TransactionDT', 'TransactionAmt', 'hour', 'day', 'day_of_month'],
    'Card_features': [c for c in train.columns if c.startswith('card')],
}

# 遍历不同特征集进行PCA和t-SNE
for name, cols in feature_sets.items():
    if len(cols) < 2:
        print(f"Skipping {name} - not enough features")
        continue
    
    print(f"\n--- {name}: {len(cols)} features ---")
    
    # 提取特征
    X_sample = train_scaled[cols].iloc[sample_idx].values
    
    # 处理NaN
    X_sample = np.nan_to_num(X_sample, nan=0)
    
    # PCA
    pca = PCA(n_components=min(10, len(cols)), random_state=42)
    pca_result = pca.fit_transform(X_sample)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    # 可视化PCA
    plt.figure(figsize=(10, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                c=y_sample, cmap='coolwarm', alpha=0.5, s=10)
    plt.colorbar(label='isFraud')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'PCA - {name}')
    plt.savefig(f'../result/pca_{name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # t-SNE (先用PCA降到20维)
    n_pca = min(20, len(cols))
    pca_tsne = PCA(n_components=n_pca, random_state=42)
    X_pca = pca_tsne.fit_transform(X_sample)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    tsne_result = tsne.fit_transform(X_pca)
    
    # 可视化t-SNE
    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                c=y_sample, cmap='coolwarm', alpha=0.5, s=10)
    plt.colorbar(label='isFraud')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title(f't-SNE - {name}')
    plt.savefig(f'../result/tsne_{name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: pca_{name}.png, tsne_{name}.png")

# ========================
# 6. 综合特征PCA（V + C + D + Time）
# ========================
print("\n=== Combined Features PCA ===")

combined_cols = (feature_sets['V_features'] + 
                feature_sets['C_features'] + 
                feature_sets['D_features'] + 
                feature_sets['Time_Amt'])

X_combined = train_scaled[combined_cols].iloc[sample_idx].values
X_combined = np.nan_to_num(X_combined, nan=0)

# PCA
pca_combined = PCA(n_components=10, random_state=42)
pca_result_combined = pca_combined.fit_transform(X_combined)
print(f"Combined PCA explained variance: {pca_combined.explained_variance_ratio_.sum():.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(pca_result_combined[:, 0], pca_result_combined[:, 1], 
            c=y_sample, cmap='coolwarm', alpha=0.5, s=10)
plt.colorbar(label='isFraud')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA - Combined (V+C+D+Time)')
plt.savefig('../result/pca_combined.png', dpi=150, bbox_inches='tight')
plt.close()

# t-SNE
pca_50 = PCA(n_components=20, random_state=42)
X_pca_combined = pca_50.fit_transform(X_combined)

tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
tsne_result_combined = tsne.fit_transform(X_pca_combined)

plt.figure(figsize=(10, 6))
plt.scatter(tsne_result_combined[:, 0], tsne_result_combined[:, 1], 
            c=y_sample, cmap='coolwarm', alpha=0.5, s=10)
plt.colorbar(label='isFraud')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE - Combined (V+C+D+Time)')
plt.savefig('../result/tsne_combined.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nAll visualizations saved!")

# ========================
# 6. 保存处理后的数据
# ========================
print("\n=== Saving Processed Data ===")

# 转换为numpy
X_train = train_scaled.values
X_test = test_scaled.values

# 保存
np.save('../resources/X_train_processed.npy', X_train)
np.save('../resources/y_train.npy', y)
np.save('../resources/X_test_processed.npy', X_test)
np.save('../resources/test_ids.npy', test_ids)

# 保存特征名
with open('../resources/feature_names.txt', 'w') as f:
    f.write('\n'.join(train_scaled.columns.tolist()))

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print("\nAll processed data saved!")
print("\nFeature engineering and dimensionality reduction completed!")
