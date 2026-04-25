import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from collections import Counter
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--train',  type=str, default='/Users/renyihuang/NLP/train_data.csv')
parser.add_argument('--test',   type=str, default='/Users/renyihuang/NLP/test_data.csv')
parser.add_argument('--output', type=str, default='/Users/renyihuang/NLP/submission_v4.csv')
args = parser.parse_args()

train = pd.read_csv(args.train)
test  = pd.read_csv(args.test)
print("資料載入完成: train={}, test={}".format(len(train), len(test)))

# ============================================================
# 解析 word_id
# ============================================================
def get_word_key(wid):
    parts = wid.split('_')
    try:
        page_idx = parts.index('page')
        text = '_'.join(parts[:page_idx - 1])
        page = parts[page_idx + 1]
        widx = parts[page_idx + 2]
        return '{}_page_{}_{}'.format(text, page, widx)
    except:
        return wid

train['word_key'] = train['word_id'].apply(get_word_key)
test['word_key']  = test['word_id'].apply(get_word_key)

STRIP_CHARS = ".,;:!?()[]{}\"'-–—«»„\""
train['word_clean'] = train['word'].str.strip(STRIP_CHARS)
test['word_clean']  = test['word'].str.strip(STRIP_CHARS)
train['word_lower'] = train['word_clean'].str.lower()
test['word_lower']  = test['word_clean'].str.lower()
train['text_type']  = train['text'].str.split('_').str[0]
test['text_type']   = test['text'].str.split('_').str[0]

# ============================================================
# 查找表
# ============================================================
participant_stats = train.groupby('participant_id')['answer'].agg(['mean','median','std'])
participant_stats.columns = ['part_mean','part_median','part_std']

# 參與者跳過率
part_skip = train.groupby('participant_id').apply(lambda x: (x['answer'] == 0).mean()).rename('part_skip_rate')
participant_stats = participant_stats.join(part_skip)

# 參與者 × 文本類型的平均TRT
part_texttype = train.groupby(['participant_id','text_type'])['answer'].mean().rename('part_tt_mean').reset_index()

word_form_stats = train.groupby('word_clean')['answer'].agg(['mean','median','std','count'])
word_form_stats.columns = ['wf_mean','wf_median','wf_std','wf_count']

word_lower_stats = train.groupby('word_lower')['answer'].agg(['mean','median','count'])
word_lower_stats.columns = ['wl_mean','wl_median','wl_count']

text_type_stats = train.groupby('text_type')['answer'].agg(['mean','std'])
text_type_stats.columns = ['tt_mean','tt_std']

word_key_stats = train.groupby('word_key')['answer'].agg(['mean','median','std'])
word_key_stats.columns = ['wk_mean','wk_median','wk_std']

# 前後詞長度
train['word_len_raw'] = train['word_clean'].str.len().fillna(0)
train_sorted = train.drop_duplicates('word_key').sort_values(['text','word_key'])
train_sorted['prev_word_len']  = train_sorted.groupby('text')['word_len_raw'].shift(1).fillna(0)
train_sorted['next_word_len']  = train_sorted.groupby('text')['word_len_raw'].shift(-1).fillna(0)
train_sorted['prev2_word_len'] = train_sorted.groupby('text')['word_len_raw'].shift(2).fillna(0)
train_sorted['next2_word_len'] = train_sorted.groupby('text')['word_len_raw'].shift(-2).fillna(0)
neighbor_stats = train_sorted[['word_key','prev_word_len','next_word_len','prev2_word_len','next2_word_len']]
train = train.merge(neighbor_stats, on='word_key', how='left')

# 文本層級統計（每個 text 共幾個詞，位置比例）
text_len_map = train.drop_duplicates('word_key').groupby('text')['word_key'].count().rename('text_n_words')
train = train.merge(text_len_map, on='text', how='left')
test  = test.merge(text_len_map, on='text', how='left')

# n-gram 頻率
all_words = train['word_lower'].fillna('').tolist()
bigram_counter  = Counter()
trigram_counter = Counter()
for w in all_words:
    for i in range(len(w)-1):
        bigram_counter[w[i:i+2]]  += 1
    for i in range(len(w)-2):
        trigram_counter[w[i:i+3]] += 1
total_bigrams  = sum(bigram_counter.values())  + 1
total_trigrams = sum(trigram_counter.values()) + 1

def avg_bigram_freq(word):
    w = str(word).lower()
    if len(w) < 2: return 0.0
    return float(np.mean([bigram_counter.get(w[i:i+2], 0) / total_bigrams for i in range(len(w)-1)]))

def avg_trigram_freq(word):
    w = str(word).lower()
    if len(w) < 3: return 0.0
    return float(np.mean([trigram_counter.get(w[i:i+3], 0) / total_trigrams for i in range(len(w)-2)]))

word_skip_rate   = train.groupby('word_lower').apply(lambda x: (x['answer'] == 0).mean()).rename('skip_rate')
nonzero_train    = train[train['answer'] > 0]
word_nonzero_mean = nonzero_train.groupby('word_lower')['answer'].mean().rename('wl_nonzero_mean')

global_mean         = train['answer'].mean()
global_median       = train['answer'].median()
global_std          = train['answer'].std()
global_skip_rate    = (train['answer'] == 0).mean()
global_nonzero_mean = nonzero_train['answer'].mean()

print("查找表建立完成 | 全局跳過率: {:.1%}".format(global_skip_rate))

# ============================================================
# 特徵工程
# ============================================================
VOWELS = set('aeiouAEIOUăâîĂÂÎ')
RO_DIAC = set('ăâîșțĂÂÎȘȚ')
CONSONANT_CLUSTERS = ['str','ntr','mpr','ngr','ndr','spr','scr','zdr','zgr']
RO_STOPWORDS = {'de','la','în','și','cu','pe','că','ca','sa','se','nu','dar','sau','un','o','al','ale','lui','ei','le','îi','ne','vă','mă','te'}

def engineer(df, is_train=True):
    df = df.copy()
    wc = df['word_clean'].fillna('').astype(str)
    wl = df['word_lower'].fillna('').astype(str)

    df['word_len']       = wc.str.len().astype(float)
    df['n_vowels']       = wc.apply(lambda w: sum(1 for c in w if c in VOWELS))
    df['n_consonants']   = df['word_len'] - df['n_vowels']
    df['vowel_ratio']    = df['n_vowels'] / (df['word_len'] + 1)
    df['has_diacritics'] = wc.apply(lambda w: int(any(c in RO_DIAC for c in w)))
    df['n_diacritics']   = wc.apply(lambda w: sum(1 for c in w if c in RO_DIAC))
    df['is_numeric']     = wc.apply(lambda w: int(w.replace('.','').replace(',','').isdigit() and len(w) > 0))
    df['is_url']         = df['word'].fillna('').apply(lambda w: int('http' in w or 'www.' in w))
    df['is_capitalized'] = wc.apply(lambda w: int(len(w) > 0 and w[0].isupper()))
    df['is_all_caps']    = wc.apply(lambda w: int(len(w) > 1 and w.isupper()))
    df['word_len_sq']    = df['word_len'] ** 2
    df['word_len_log']   = np.log1p(df['word_len'])
    df['word_len_sqrt']  = np.sqrt(df['word_len'])
    df['has_hyphen']     = df['word'].fillna('').apply(lambda w: int('-' in w))
    df['has_cluster']    = wl.apply(lambda w: int(any(c in w for c in CONSONANT_CLUSTERS)))
    df['is_stopword']    = wl.apply(lambda w: int(w in RO_STOPWORDS))

    df['bigram_freq']     = wl.apply(avg_bigram_freq)
    df['trigram_freq']    = wl.apply(avg_trigram_freq)
    df['bigram_freq_log'] = np.log1p(df['bigram_freq'] * 1000)

    df['word_pos']      = df['word_key'].str.split('_').str[-1].astype(float)
    df['page_num']      = df['word_key'].str.split('_').apply(lambda x: float(x[-2]) if len(x) >= 2 else 0.0)
    df['word_pos_log']  = np.log1p(df['word_pos'])
    df['word_pos_sq']   = df['word_pos'] ** 2

    # 位置比例（在整個文本中的相對位置）
    df['text_n_words']  = df['text_n_words'].fillna(df['word_pos'] + 1)
    df['pos_ratio']     = df['word_pos'] / (df['text_n_words'] + 1)
    df['is_last_10pct'] = (df['pos_ratio'] > 0.9).astype(float)
    df['is_first_10pct']= (df['pos_ratio'] < 0.1).astype(float)

    # 參與者特徵
    df = df.merge(participant_stats, on='participant_id', how='left')
    df['part_mean']      = df['part_mean'].fillna(global_mean)
    df['part_median']    = df['part_median'].fillna(global_median)
    df['part_std']       = df['part_std'].fillna(global_std)
    df['part_skip_rate'] = df['part_skip_rate'].fillna(global_skip_rate)
    df['part_speed_rel'] = df['part_mean'] / global_mean

    # 參與者 × 文本類型
    df = df.merge(part_texttype, on=['participant_id','text_type'], how='left')
    df['part_tt_mean']   = df['part_tt_mean'].fillna(df['part_mean'])
    df['part_tt_rel']    = df['part_tt_mean'] / (global_mean + 1e-9)

    # 單詞查找
    df = df.merge(word_form_stats, on='word_clean', how='left')
    df['wf_mean']   = df['wf_mean'].fillna(global_mean)
    df['wf_median'] = df['wf_median'].fillna(global_median)
    df['wf_std']    = df['wf_std'].fillna(global_std)
    df['wf_count']  = df['wf_count'].fillna(0)

    df = df.merge(word_lower_stats, on='word_lower', how='left')
    df['wl_mean']   = df['wl_mean'].fillna(df['wf_mean'])
    df['wl_median'] = df['wl_median'].fillna(df['wf_median'])
    df['wl_count']  = df['wl_count'].fillna(0)

    df = df.merge(text_type_stats, on='text_type', how='left')
    df['tt_mean'] = df['tt_mean'].fillna(global_mean)
    df['tt_std']  = df['tt_std'].fillna(global_std)

    df = df.merge(word_key_stats, on='word_key', how='left')
    df['wk_mean']   = df['wk_mean'].fillna(df['wf_mean'])
    df['wk_median'] = df['wk_median'].fillna(df['wf_median'])
    df['wk_std']    = df['wk_std'].fillna(df['wf_std'])

    df = df.merge(word_skip_rate, on='word_lower', how='left')
    df['skip_rate'] = df['skip_rate'].fillna(global_skip_rate)

    df = df.merge(word_nonzero_mean, on='word_lower', how='left')
    df['wl_nonzero_mean'] = df['wl_nonzero_mean'].fillna(global_nonzero_mean)

    for col in ['prev_word_len','next_word_len','prev2_word_len','next2_word_len']:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0)
    df['neighbor_len_sum']  = df['prev_word_len'] + df['next_word_len']
    df['neighbor_len_mean'] = df['neighbor_len_sum'] / 2

    # 互動特徵
    df['wf_x_part']      = df['wf_mean'] * df['part_mean'] / global_mean
    df['wk_x_part']      = df['wk_mean'] * df['part_mean'] / global_mean
    df['len_x_part']     = df['word_len'] * df['part_mean']
    df['wf_seen']        = (df['wf_count'] > 0).astype(float)
    df['part_wf_diff']   = df['part_mean'] - df['wf_mean']
    df['skip_x_part']    = df['skip_rate'] * df['part_mean']
    df['bigram_x_len']   = df['bigram_freq'] * df['word_len']
    df['expected_trt']   = df['wl_nonzero_mean'] * (1 - df['skip_rate'])
    df['expected_x_part']= df['expected_trt'] * df['part_speed_rel']
    # 參與者跳過率 vs 詞跳過率
    df['skip_diff']      = df['part_skip_rate'] - df['skip_rate']
    # 詞長 × 文本類型均值
    df['len_x_tt']       = df['word_len'] * df['tt_mean']
    # 參與者 × 文本類型互動
    df['part_tt_x_wf']   = df['part_tt_mean'] * df['wf_mean'] / global_mean

    return df

print("開始特徵工程...")
train = engineer(train, is_train=True)
test  = engineer(test,  is_train=False)
print("特徵工程完成")

# ============================================================
# 特徵列表
# ============================================================
FEATURES = [
    'word_len','n_vowels','n_consonants','vowel_ratio',
    'has_diacritics','n_diacritics','is_numeric','is_url',
    'is_capitalized','is_all_caps','has_hyphen','has_cluster','is_stopword',
    'word_len_sq','word_len_log','word_len_sqrt',
    'bigram_freq','trigram_freq','bigram_freq_log',
    'word_pos','page_num','word_pos_log','word_pos_sq',
    'pos_ratio','is_last_10pct','is_first_10pct','text_n_words',
    'prev_word_len','next_word_len','prev2_word_len','next2_word_len',
    'neighbor_len_sum','neighbor_len_mean',
    'part_mean','part_median','part_std','part_speed_rel',
    'part_skip_rate','part_tt_mean','part_tt_rel',
    'wf_mean','wf_median','wf_std','wf_count',
    'wl_mean','wl_median','wl_count',
    'wk_mean','wk_median','wk_std',
    'skip_rate','wl_nonzero_mean',
    'tt_mean','tt_std',
    'wf_x_part','wk_x_part','len_x_part','wf_seen',
    'part_wf_diff','skip_x_part','bigram_x_len',
    'expected_trt','expected_x_part',
    'skip_diff','len_x_tt','part_tt_x_wf',
]

X      = train[FEATURES].fillna(0).values
y      = train['answer'].values
X_test = test[FEATURES].fillna(0).values
print("特徵數量:", len(FEATURES))

# ============================================================
# 評分函數
# ============================================================
def eval_metric(y_true, preds):
    y_true = np.array(y_true, dtype=float)
    preds  = np.array(preds,  dtype=float)
    r2     = max(0.0, r2_score(y_true, preds))
    pears  = pearsonr(y_true, preds)[0]
    if np.isnan(pears): pears = 0.0
    return 100.0 * (abs(pears) + r2) / 2.0

# ============================================================
# 模型定義
# ============================================================
lgb_params = dict(
    n_estimators=2000, learning_rate=0.02, num_leaves=127,
    min_child_samples=15, reg_alpha=0.05, reg_lambda=0.05,
    subsample=0.8, colsample_bytree=0.8,
    n_jobs=-1, random_state=42, verbose=-1,
)
modelA = lgb.LGBMRegressor(**lgb_params)

modelB = HistGradientBoostingRegressor(
    max_iter=1000, learning_rate=0.02, max_leaf_nodes=127,
    min_samples_leaf=15, l2_regularization=0.05,
    max_bins=255, random_state=42
)
modelC = ExtraTreesRegressor(
    n_estimators=300, max_depth=15, min_samples_leaf=10,
    n_jobs=-1, random_state=42
)
modelD = RandomForestRegressor(
    n_estimators=300, max_depth=15, min_samples_leaf=10,
    n_jobs=-1, random_state=42
)

# ============================================================
# 交叉驗證 + Stacking
# ============================================================
groups = train['text'].values
gkf    = GroupKFold(n_splits=9)

oof_A = np.zeros(len(X))
oof_B = np.zeros(len(X))
oof_C = np.zeros(len(X))
oof_D = np.zeros(len(X))
cv_scores = []

print("=" * 65)
print("交叉驗證")
print("=" * 65)

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    modelA.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
               callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    modelB.fit(X_tr, y_tr)
    modelC.fit(X_tr, y_tr)
    modelD.fit(X_tr, y_tr)

    pA = modelA.predict(X_val)
    pB = modelB.predict(X_val)
    pC = modelC.predict(X_val)
    pD = modelD.predict(X_val)

    oof_A[val_idx] = pA
    oof_B[val_idx] = pB
    oof_C[val_idx] = pC
    oof_D[val_idx] = pD

    p_ens = 0.50 * pA + 0.25 * pB + 0.15 * pC + 0.10 * pD

    sA   = eval_metric(y_val, pA)
    sB   = eval_metric(y_val, pB)
    sC   = eval_metric(y_val, pC)
    sD   = eval_metric(y_val, pD)
    sEns = eval_metric(y_val, p_ens)
    cv_scores.append(sEns)

    g = groups[val_idx][0]
    print("Fold {:d} ({:20s}) | A:{:.1f} B:{:.1f} C:{:.1f} D:{:.1f} | Ens:{:.1f}".format(
        fold+1, g, sA, sB, sC, sD, sEns))

print("=" * 65)
print("平均 Ensemble: {:.2f} +/- {:.2f}".format(np.mean(cv_scores), np.std(cv_scores)))
print("=" * 65)

# ============================================================
# Meta-model Stacking（用 LightGBM 當 meta-model）
# ============================================================
print("\nMeta-model Stacking 訓練中...")
X_meta_train = np.column_stack([oof_A, oof_B, oof_C, oof_D])

meta_lgb_params = dict(
    n_estimators=500, learning_rate=0.05, num_leaves=15,
    min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
    n_jobs=-1, random_state=42, verbose=-1,
)
meta_model = lgb.LGBMRegressor(**meta_lgb_params)
meta_model.fit(X_meta_train, y)

meta_oof = meta_model.predict(X_meta_train)
print("Meta-model OOF score: {:.2f}".format(eval_metric(y, meta_oof)))

# ============================================================
# 全部資料重新訓練
# ============================================================
print("\n全部資料重新訓練中...")
modelA.fit(X, y)
modelB.fit(X, y)
modelC.fit(X, y)
modelD.fit(X, y)

pA_test = modelA.predict(X_test)
pB_test = modelB.predict(X_test)
pC_test = modelC.predict(X_test)
pD_test = modelD.predict(X_test)

X_meta_test = np.column_stack([pA_test, pB_test, pC_test, pD_test])
meta_preds  = meta_model.predict(X_meta_test)

p_ens_test  = 0.50 * pA_test + 0.25 * pB_test + 0.15 * pC_test + 0.10 * pD_test
final_preds = 0.6 * p_ens_test + 0.4 * meta_preds
final_preds = np.clip(final_preds, 0, None)

# ============================================================
# 儲存結果
# ============================================================
out = test[['datapointID']].copy()
out['subtaskID'] = 1
out['answer']    = final_preds
out = out[['subtaskID', 'datapointID', 'answer']]
out.to_csv(args.output, index=False)

print("\n完成！")
print("輸出前10行:")
print(out.head(10).to_string())
print("\n檔案已存到:", args.output)
