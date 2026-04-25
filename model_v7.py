"""
model_v7.py — Anti-leakage version
Fix: all target-dependent statistics computed inside each CV fold
     from training split only. Test uses full training data (correct).
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from collections import Counter
import lightgbm as lgb
import xgboost as xgb
from wordfreq import zipf_frequency, word_frequency
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--train',  default='/Users/renyihuang/NLP/train_data.csv')
parser.add_argument('--test',   default='/Users/renyihuang/NLP/test_data.csv')
parser.add_argument('--output', default='/Users/renyihuang/NLP/submission_v7.csv')
args = parser.parse_args()

# ============================================================
# Load & parse
# ============================================================
raw_train = pd.read_csv(args.train)
raw_test  = pd.read_csv(args.test)
print("Loaded: train={}, test={}".format(len(raw_train), len(raw_test)))

STRIP_CHARS = ".,;:!?()[]{}\"'-–—«»„\""

def parse_base(df):
    df = df.copy()
    def get_word_key(wid):
        parts = wid.split('_')
        try:
            pi = parts.index('page')
            return '{}_page_{}_{}'.format('_'.join(parts[:pi-1]), parts[pi+1], parts[pi+2])
        except:
            return wid
    df['word_key']   = df['word_id'].apply(get_word_key)
    df['word_clean'] = df['word'].str.strip(STRIP_CHARS)
    df['word_lower'] = df['word_clean'].str.lower()
    df['text_type']  = df['text'].str.split('_').str[0]
    return df

raw_train = parse_base(raw_train)
raw_test  = parse_base(raw_test)

# ============================================================
# Non-target features (no leakage — external data or structure only)
# ============================================================
VOWELS = set('aeiouAEIOUăâîĂÂÎ')
RO_DIAC = set('ăâîșțĂÂÎȘȚ')
CONSONANT_CLUSTERS = ['str','ntr','mpr','ngr','ndr','spr','scr','zdr','zgr']
RO_STOPWORDS = {
    'de','la','în','și','cu','pe','că','ca','sa','se','nu','dar','sau',
    'un','o','al','ale','lui','ei','le','îi','ne','vă','mă','te','este',
    'sunt','era','ai','am','avem','fi','fost','care','ce','din','prin',
    'mai','poate','când','cum','tot','toate','acest','această','aceste'
}

# wordfreq cache (external corpus — no leakage)
all_words = set(raw_train['word_lower'].dropna()) | set(raw_test['word_lower'].dropna())
print("Building wordfreq cache for {} unique words...".format(len(all_words)))
zipf_ro_cache  = {w: zipf_frequency(w, 'ro') for w in all_words}
zipf_en_cache  = {w: zipf_frequency(w, 'en') for w in all_words}
freq_ro_cache  = {w: word_frequency(w, 'ro')  for w in all_words}

# text word-count map (structure only, not target)
text_len_map = (raw_train.drop_duplicates('word_key')
                .groupby('text')['word_key'].count()
                .rename('text_n_words'))

def add_structural_features(df):
    """Features that don't use answer labels — safe to compute globally."""
    df = df.copy()
    wc = df['word_clean'].fillna('').astype(str)
    wl = df['word_lower'].fillna('').astype(str)

    df['word_len']       = wc.str.len().astype(float)
    df['n_vowels']       = wc.apply(lambda w: sum(1 for c in w if c in VOWELS))
    df['n_consonants']   = df['word_len'] - df['n_vowels']
    df['vowel_ratio']    = df['n_vowels'] / (df['word_len'] + 1)
    df['has_diacritics'] = wc.apply(lambda w: int(any(c in RO_DIAC for c in w)))
    df['n_diacritics']   = wc.apply(lambda w: sum(1 for c in w if c in RO_DIAC))
    df['is_numeric']     = wc.apply(lambda w: int(w.replace('.','').replace(',','').isdigit() and len(w)>0))
    df['is_url']         = df['word'].fillna('').apply(lambda w: int('http' in w or 'www.' in w))
    df['is_capitalized'] = wc.apply(lambda w: int(len(w)>0 and w[0].isupper()))
    df['is_all_caps']    = wc.apply(lambda w: int(len(w)>1 and w.isupper()))
    df['word_len_sq']    = df['word_len'] ** 2
    df['word_len_log']   = np.log1p(df['word_len'])
    df['word_len_sqrt']  = np.sqrt(df['word_len'])
    df['has_hyphen']     = df['word'].fillna('').apply(lambda w: int('-' in w))
    df['has_cluster']    = wl.apply(lambda w: int(any(c in w for c in CONSONANT_CLUSTERS)))
    df['is_stopword']    = wl.apply(lambda w: int(w in RO_STOPWORDS))
    df['n_unique_chars'] = wc.apply(lambda w: len(set(w.lower())))
    df['char_diversity'] = df['n_unique_chars'] / (df['word_len'] + 1)

    # wordfreq (external corpus)
    df['zipf_ro']      = wl.map(zipf_ro_cache).fillna(0.0)
    df['zipf_en']      = wl.map(zipf_en_cache).fillna(0.0)
    df['zipf_max']     = df[['zipf_ro','zipf_en']].max(axis=1)
    df['freq_log']     = np.log1p(wl.map(freq_ro_cache).fillna(0.0) * 1e6)
    df['is_unknown_ro']= (df['zipf_ro'] == 0).astype(float)
    df['is_rare']      = (df['zipf_ro'] < 2).astype(float)
    df['is_common']    = (df['zipf_ro'] > 4).astype(float)
    df['zipf_x_len']   = df['zipf_ro'] * df['word_len']

    # position
    df['word_pos']     = df['word_key'].str.split('_').str[-1].astype(float)
    df['page_num']     = df['word_key'].str.split('_').apply(lambda x: float(x[-2]) if len(x)>=2 else 0.0)
    df['word_pos_log'] = np.log1p(df['word_pos'])
    df['word_pos_sq']  = df['word_pos'] ** 2
    df = df.merge(text_len_map, on='text', how='left')
    df['text_n_words'] = df['text_n_words'].fillna(df['word_pos'] + 1)
    df['pos_ratio']    = df['word_pos'] / (df['text_n_words'] + 1)
    df['is_first_10pct']= (df['pos_ratio'] < 0.1).astype(float)
    df['is_last_10pct'] = (df['pos_ratio'] > 0.9).astype(float)

    return df

raw_train = add_structural_features(raw_train)
raw_test  = add_structural_features(raw_test)

# ============================================================
# Target-dependent lookup tables — computed per fold
# ============================================================
def compute_target_tables(df):
    """All statistics that depend on answer labels."""
    nz = df[df['answer'] > 0]
    tables = {}
    tables['global_mean']         = df['answer'].mean()
    tables['global_median']       = df['answer'].median()
    tables['global_std']          = df['answer'].std()
    tables['global_skip_rate']    = (df['answer'] == 0).mean()
    tables['global_nonzero_mean'] = nz['answer'].mean() if len(nz) else df['answer'].mean()
    tables['global_nonzero_std']  = nz['answer'].std()  if len(nz) else df['answer'].std()

    p = df.groupby('participant_id')['answer'].agg(['mean','median','std'])
    p.columns = ['part_mean','part_median','part_std']
    p['part_skip_rate']    = df.groupby('participant_id').apply(lambda x: (x['answer']==0).mean())
    p['part_nonzero_mean'] = nz.groupby('participant_id')['answer'].mean() if len(nz) else np.nan
    tables['participant_stats'] = p

    pt = df.groupby(['participant_id','text_type'])['answer'].mean().rename('part_tt_mean').reset_index()
    ps = df.groupby(['participant_id','text_type']).apply(lambda x: (x['answer']==0).mean()).rename('part_tt_skip').reset_index()
    tables['part_texttype'] = pt
    tables['part_tt_skip']  = ps

    wf = df.groupby('word_clean')['answer'].agg(['mean','median','std','count'])
    wf.columns = ['wf_mean','wf_median','wf_std','wf_count']
    tables['word_form_stats'] = wf

    wl = df.groupby('word_lower')['answer'].agg(['mean','median','std','count'])
    wl.columns = ['wl_mean','wl_median','wl_std','wl_count']
    tables['word_lower_stats'] = wl

    wk = df.groupby('word_key')['answer'].agg(['mean','median','std','count'])
    wk.columns = ['wk_mean','wk_median','wk_std','wk_count']
    tables['word_key_stats'] = wk

    tx = df.groupby('text')['answer'].agg(['mean','std','median'])
    tx.columns = ['text_mean','text_std','text_median']
    tables['text_stats'] = tx

    tt = df.groupby('text_type')['answer'].agg(['mean','std'])
    tt.columns = ['tt_mean','tt_std']
    tables['text_type_stats'] = tt

    sr = df.groupby('word_lower').apply(lambda x: (x['answer']==0).mean()).rename('skip_rate')
    tables['word_skip_rate'] = sr

    nz_mean = nz.groupby('word_lower')['answer'].mean().rename('wl_nonzero_mean') if len(nz) else pd.Series(dtype=float)
    nz_std  = nz.groupby('word_lower')['answer'].std().rename('wl_nonzero_std')  if len(nz) else pd.Series(dtype=float)
    tables['word_nonzero_mean'] = nz_mean
    tables['word_nonzero_std']  = nz_std

    # n-gram from this fold's words only
    words = df['word_lower'].fillna('').tolist()
    bg, tg = Counter(), Counter()
    for w in words:
        for i in range(len(w)-1): bg[w[i:i+2]] += 1
        for i in range(len(w)-2): tg[w[i:i+3]] += 1
    tbg = sum(bg.values()) + 1
    ttg = sum(tg.values()) + 1
    tables['bigram_counter']  = bg
    tables['trigram_counter'] = tg
    tables['total_bigrams']   = tbg
    tables['total_trigrams']  = ttg

    # neighbor word lengths
    df2 = df.copy()
    df2['word_len_raw'] = df2['word_clean'].str.len().fillna(0)
    srt = df2.drop_duplicates('word_key').sort_values(['text','word_key'])
    for sh, nm in [(1,'prev'),(-1,'next'),(2,'prev2'),(-2,'next2')]:
        srt[f'{nm}_word_len'] = srt.groupby('text')['word_len_raw'].shift(sh).fillna(0)
    tables['neighbor_stats'] = srt[['word_key','prev_word_len','next_word_len','prev2_word_len','next2_word_len']]

    return tables

def apply_target_features(df, T):
    """Merge target-dependent features using precomputed tables T."""
    df = df.copy()
    gm  = T['global_mean']
    gmd = T['global_median']
    gsd = T['global_std']
    gsr = T['global_skip_rate']
    gnm = T['global_nonzero_mean']
    gns = T['global_nonzero_std']

    bg = T['bigram_counter']; tbg = T['total_bigrams']
    tg = T['trigram_counter']; ttg = T['total_trigrams']
    wl_col = df['word_lower'].fillna('').astype(str)

    def avg_bg(word):
        w = str(word).lower()
        if len(w) < 2: return 0.0
        return float(np.mean([bg.get(w[i:i+2],0)/tbg for i in range(len(w)-1)]))
    def avg_tg(word):
        w = str(word).lower()
        if len(w) < 3: return 0.0
        return float(np.mean([tg.get(w[i:i+3],0)/ttg for i in range(len(w)-2)]))

    df['bigram_freq']      = wl_col.apply(avg_bg)
    df['trigram_freq']     = wl_col.apply(avg_tg)
    df['bigram_freq_log']  = np.log1p(df['bigram_freq'] * 1000)
    df['trigram_freq_log'] = np.log1p(df['trigram_freq'] * 1000)

    # participant
    df = df.merge(T['participant_stats'], on='participant_id', how='left')
    df['part_mean']         = df['part_mean'].fillna(gm)
    df['part_median']       = df['part_median'].fillna(gmd)
    df['part_std']          = df['part_std'].fillna(gsd)
    df['part_skip_rate']    = df['part_skip_rate'].fillna(gsr)
    df['part_nonzero_mean'] = df['part_nonzero_mean'].fillna(gnm)
    df['part_speed_rel']    = df['part_mean'] / gm

    df = df.merge(T['part_texttype'], on=['participant_id','text_type'], how='left')
    df['part_tt_mean'] = df['part_tt_mean'].fillna(df['part_mean'])
    df['part_tt_rel']  = df['part_tt_mean'] / (gm + 1e-9)

    df = df.merge(T['part_tt_skip'], on=['participant_id','text_type'], how='left')
    df['part_tt_skip'] = df['part_tt_skip'].fillna(df['part_skip_rate'])

    # word form
    df = df.merge(T['word_form_stats'], on='word_clean', how='left')
    df['wf_mean']   = df['wf_mean'].fillna(gm)
    df['wf_median'] = df['wf_median'].fillna(gmd)
    df['wf_std']    = df['wf_std'].fillna(gsd)
    df['wf_count']  = df['wf_count'].fillna(0)

    # word lower
    df = df.merge(T['word_lower_stats'], on='word_lower', how='left')
    df['wl_mean']   = df['wl_mean'].fillna(df['wf_mean'])
    df['wl_median'] = df['wl_median'].fillna(df['wf_median'])
    df['wl_std']    = df['wl_std'].fillna(df['wf_std'])
    df['wl_count']  = df['wl_count'].fillna(0)

    # word key
    df = df.merge(T['word_key_stats'], on='word_key', how='left')
    df['wk_mean']   = df['wk_mean'].fillna(df['wf_mean'])
    df['wk_median'] = df['wk_median'].fillna(df['wf_median'])
    df['wk_std']    = df['wk_std'].fillna(df['wf_std'])
    df['wk_count']  = df['wk_count'].fillna(0)

    # text
    df = df.merge(T['text_stats'], on='text', how='left')
    df['text_mean']   = df['text_mean'].fillna(gm)
    df['text_std']    = df['text_std'].fillna(gsd)
    df['text_median'] = df['text_median'].fillna(gmd)

    # text type
    df = df.merge(T['text_type_stats'], on='text_type', how='left')
    df['tt_mean'] = df['tt_mean'].fillna(gm)
    df['tt_std']  = df['tt_std'].fillna(gsd)

    # skip rate & nonzero mean
    df = df.merge(T['word_skip_rate'], on='word_lower', how='left')
    df['skip_rate'] = df['skip_rate'].fillna(gsr)

    df = df.merge(T['word_nonzero_mean'], on='word_lower', how='left')
    df['wl_nonzero_mean'] = df['wl_nonzero_mean'].fillna(gnm)
    df = df.merge(T['word_nonzero_std'], on='word_lower', how='left')
    df['wl_nonzero_std']  = df['wl_nonzero_std'].fillna(gns)

    # neighbors
    df = df.merge(T['neighbor_stats'], on='word_key', how='left')
    for col in ['prev_word_len','next_word_len','prev2_word_len','next2_word_len']:
        df[col] = df[col].fillna(0)
    df['neighbor_len_sum']  = df['prev_word_len'] + df['next_word_len']
    df['neighbor_len_mean'] = df['neighbor_len_sum'] / 2
    df['neighbor_len_max']  = df[['prev_word_len','next_word_len']].max(axis=1)

    # interaction features
    df['wf_x_part']       = df['wf_mean'] * df['part_mean'] / gm
    df['wk_x_part']       = df['wk_mean'] * df['part_mean'] / gm
    df['len_x_part']      = df['word_len'] * df['part_mean']
    df['wf_seen']         = (df['wf_count'] > 0).astype(float)
    df['wk_seen']         = (df['wk_count'] > 0).astype(float)
    df['part_wf_diff']    = df['part_mean'] - df['wf_mean']
    df['skip_x_part']     = df['skip_rate'] * df['part_mean']
    df['bigram_x_len']    = df['bigram_freq'] * df['word_len']
    df['expected_trt']    = df['wl_nonzero_mean'] * (1 - df['skip_rate'])
    df['expected_x_part'] = df['expected_trt'] * df['part_speed_rel']
    df['skip_diff']       = df['part_skip_rate'] - df['skip_rate']
    df['wk_vs_wf']        = df['wk_mean'] - df['wf_mean']
    df['part_vs_text']    = df['part_mean'] - df['text_mean']
    df['nz_vs_part']      = df['wl_nonzero_mean'] / (df['part_nonzero_mean'] + 1e-9)
    df['zipf_x_part']     = df['zipf_max'] * df['part_speed_rel']
    df['zipf_x_skip']     = df['zipf_max'] * (1 - df['skip_rate'])
    df['rare_x_part']     = df['is_rare'] * df['part_mean']
    df['skip_x_skip']     = df['part_skip_rate'] * df['skip_rate']
    df['word_difficulty'] = df['word_len'] * (1 - df['bigram_freq'] * 100)

    return df

FEATURES = [
    'word_len','n_vowels','n_consonants','vowel_ratio',
    'has_diacritics','n_diacritics','is_numeric','is_url',
    'is_capitalized','is_all_caps','has_hyphen','has_cluster','is_stopword',
    'word_len_sq','word_len_log','word_len_sqrt','n_unique_chars','char_diversity',
    'zipf_ro','zipf_en','zipf_max','freq_log',
    'is_unknown_ro','is_rare','is_common','zipf_x_len',
    'bigram_freq','trigram_freq','bigram_freq_log','trigram_freq_log',
    'word_pos','page_num','word_pos_log','word_pos_sq',
    'pos_ratio','is_first_10pct','is_last_10pct','text_n_words',
    'prev_word_len','next_word_len','prev2_word_len','next2_word_len',
    'neighbor_len_sum','neighbor_len_mean','neighbor_len_max',
    'part_mean','part_median','part_std','part_speed_rel',
    'part_skip_rate','part_nonzero_mean','part_tt_mean','part_tt_rel','part_tt_skip',
    'wf_mean','wf_median','wf_std','wf_count',
    'wl_mean','wl_median','wl_std','wl_count',
    'wk_mean','wk_median','wk_std','wk_count',
    'skip_rate','wl_nonzero_mean','wl_nonzero_std',
    'text_mean','text_std','text_median','tt_mean','tt_std',
    'wf_x_part','wk_x_part','len_x_part','wf_seen','wk_seen',
    'part_wf_diff','skip_x_part','bigram_x_len',
    'expected_trt','expected_x_part','skip_diff',
    'wk_vs_wf','part_vs_text','nz_vs_part',
    'zipf_x_part','zipf_x_skip','rare_x_part','skip_x_skip','word_difficulty',
]

META_FEATS = [
    'wk_mean','wk_count','part_mean','part_skip_rate',
    'skip_rate','wl_nonzero_mean','expected_trt',
    'word_len','pos_ratio','text_mean',
    'zipf_ro','zipf_max','is_unknown_ro',
]

# ============================================================
# Metric
# ============================================================
def eval_metric(y_true, preds):
    y_true = np.array(y_true, dtype=float)
    preds  = np.array(preds,  dtype=float)
    r2    = max(0.0, r2_score(y_true, preds))
    pears = pearsonr(y_true, preds)[0]
    if np.isnan(pears): pears = 0.0
    return 100.0 * (abs(pears) + r2) / 2.0

# ============================================================
# Model params
# ============================================================
lgb_params = dict(
    n_estimators=3000, learning_rate=0.02, num_leaves=127,
    min_child_samples=15, reg_alpha=0.05, reg_lambda=0.05,
    subsample=0.8, colsample_bytree=0.8,
    n_jobs=-1, random_state=42, verbose=-1,
)
lgb_clf_params = dict(
    n_estimators=2000, learning_rate=0.02, num_leaves=63,
    min_child_samples=15, reg_alpha=0.1, reg_lambda=0.1,
    subsample=0.8, colsample_bytree=0.8,
    objective='binary', n_jobs=-1, random_state=42, verbose=-1,
)
lgb_log_params = dict(
    n_estimators=3000, learning_rate=0.02, num_leaves=127,
    min_child_samples=10, reg_alpha=0.05, reg_lambda=0.05,
    subsample=0.8, colsample_bytree=0.8,
    n_jobs=-1, random_state=43, verbose=-1,
)
xgb_params = dict(
    n_estimators=3000, learning_rate=0.02, max_depth=7,
    min_child_weight=10, reg_alpha=0.05, reg_lambda=1.0,
    subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=50,
    tree_method='hist', n_jobs=-1, random_state=42, verbosity=0,
)
hgb_params = dict(
    max_iter=1000, learning_rate=0.02, max_leaf_nodes=127,
    min_samples_leaf=15, l2_regularization=0.05,
    max_bins=255, random_state=42,
)

# ============================================================
# Cross-validation (anti-leakage: tables computed per fold)
# ============================================================
groups = raw_train['text'].values
y_all  = raw_train['answer'].values
gkf    = GroupKFold(n_splits=9)

n = len(raw_train)
oof_lgb  = np.zeros(n)
oof_xgb  = np.zeros(n)
oof_hgb  = np.zeros(n)
oof_log  = np.zeros(n)
oof_skip = np.zeros(n)
oof_meta_extra = np.zeros((n, len(META_FEATS)))
cv_scores = []

print("=" * 65)
print("Cross-Validation (anti-leakage)")
print("=" * 65)

for fold, (tr_idx, val_idx) in enumerate(gkf.split(raw_train, y_all, groups)):
    tr_df  = raw_train.iloc[tr_idx].copy()
    val_df = raw_train.iloc[val_idx].copy()

    # Compute target-dependent tables from TRAINING SPLIT ONLY
    T = compute_target_tables(tr_df)

    tr_feat  = apply_target_features(tr_df,  T)
    val_feat = apply_target_features(val_df, T)

    X_tr  = tr_feat[FEATURES].fillna(0).values
    X_val = val_feat[FEATURES].fillna(0).values
    y_tr  = tr_df['answer'].values
    y_val = val_df['answer'].values
    y_bin_tr = (y_tr == 0).astype(int)
    y_log_tr = np.where(y_tr > 0, np.log1p(y_tr), 0.0)

    cb = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]

    mA = lgb.LGBMRegressor(**lgb_params)
    mA.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=cb)
    pA = mA.predict(X_val); oof_lgb[val_idx] = pA

    mB = xgb.XGBRegressor(**xgb_params)
    mB.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    pB = mB.predict(X_val); oof_xgb[val_idx] = pB

    mC = HistGradientBoostingRegressor(**hgb_params)
    mC.fit(X_tr, y_tr)
    pC = mC.predict(X_val); oof_hgb[val_idx] = pC

    mD = lgb.LGBMClassifier(**lgb_clf_params)
    mD.fit(X_tr, y_bin_tr, eval_set=[(X_val, (y_val==0).astype(int))], callbacks=cb)
    p_skip = mD.predict_proba(X_val)[:, 1]; oof_skip[val_idx] = p_skip

    mE = lgb.LGBMRegressor(**lgb_log_params)
    y_log_val = np.where(y_val > 0, np.log1p(y_val), 0.0)
    mE.fit(X_tr, y_log_tr, eval_set=[(X_val, y_log_val)], callbacks=cb)
    p_log = np.expm1(mE.predict(X_val)); oof_log[val_idx] = p_log

    oof_meta_extra[val_idx] = val_feat[META_FEATS].fillna(0).values

    p_hurdle = (1 - p_skip) * pA
    p_ens = 0.35*pA + 0.25*pB + 0.20*pC + 0.10*p_hurdle + 0.10*p_log
    p_ens = np.clip(p_ens, 0, None)

    sA   = eval_metric(y_val, pA)
    sB   = eval_metric(y_val, pB)
    sC   = eval_metric(y_val, pC)
    sEns = eval_metric(y_val, p_ens)
    cv_scores.append(sEns)

    g = groups[val_idx][0]
    print("Fold {:d} ({:20s}) | LGB:{:.1f} XGB:{:.1f} HGB:{:.1f} | Ens:{:.1f}".format(
        fold+1, g, sA, sB, sC, sEns))

print("=" * 65)
print("CV Ensemble (honest, no leakage): {:.2f} +/- {:.2f}".format(
    np.mean(cv_scores), np.std(cv_scores)))
print("=" * 65)

# True OOF meta score
oof_hurdle = (1 - oof_skip) * oof_lgb
X_meta_oof = np.column_stack([oof_lgb, oof_xgb, oof_hgb, oof_log, oof_skip, oof_hurdle, oof_meta_extra])
print("True OOF ensemble score: {:.2f}".format(eval_metric(y_all, np.clip(oof_lgb * 0.5 + oof_xgb * 0.3 + oof_hgb * 0.2, 0, None))))

# ============================================================
# Meta-model (trained on true OOF predictions)
# ============================================================
print("\nMeta-model training on true OOF predictions...")
meta_params = dict(
    n_estimators=500, learning_rate=0.03, num_leaves=31,
    min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
    n_jobs=-1, random_state=42, verbose=-1,
)
meta_model = lgb.LGBMRegressor(**meta_params)
meta_model.fit(X_meta_oof, y_all)

# ============================================================
# Retrain on full training data for test prediction
# ============================================================
print("\nRetraining on full data...")
T_full      = compute_target_tables(raw_train)
train_full  = apply_target_features(raw_train, T_full)
test_full   = apply_target_features(raw_test,  T_full)

X_full      = train_full[FEATURES].fillna(0).values
X_test_full = test_full[FEATURES].fillna(0).values
y_full      = raw_train['answer'].values
y_bin_full  = (y_full == 0).astype(int)
y_log_full  = np.where(y_full > 0, np.log1p(y_full), 0.0)

cb = [lgb.log_evaluation(-1)]
xgb_noes = {k: v for k, v in xgb_params.items() if k != 'early_stopping_rounds'}

fA = lgb.LGBMRegressor(**lgb_params);     fA.fit(X_full, y_full, callbacks=cb)
fB = xgb.XGBRegressor(**xgb_noes);        fB.fit(X_full, y_full, verbose=False)
fC = HistGradientBoostingRegressor(**hgb_params); fC.fit(X_full, y_full)
fD = lgb.LGBMClassifier(**lgb_clf_params); fD.fit(X_full, y_bin_full, callbacks=cb)
fE = lgb.LGBMRegressor(**lgb_log_params); fE.fit(X_full, y_log_full, callbacks=cb)

pA_t    = fA.predict(X_test_full)
pB_t    = fB.predict(X_test_full)
pC_t    = fC.predict(X_test_full)
skip_t  = fD.predict_proba(X_test_full)[:, 1]
pE_t    = np.expm1(fE.predict(X_test_full))
hurdle_t = (1 - skip_t) * pA_t

X_meta_test = np.column_stack([
    pA_t, pB_t, pC_t, pE_t, skip_t, hurdle_t,
    test_full[META_FEATS].fillna(0).values
])
meta_preds  = meta_model.predict(X_meta_test)
p_ens_test  = 0.35*pA_t + 0.25*pB_t + 0.20*pC_t + 0.10*hurdle_t + 0.10*pE_t
final_preds = 0.5 * np.clip(p_ens_test, 0, None) + 0.5 * np.clip(meta_preds, 0, None)
final_preds = np.clip(final_preds, 0, None)

# ============================================================
# Save
# ============================================================
out = raw_test[['datapointID']].copy()
out['subtaskID'] = 1
out['answer']    = final_preds
out = out[['subtaskID', 'datapointID', 'answer']]
out.to_csv(args.output, index=False)

print("\nDone!")
print(out.head(10).to_string())
print("\nSaved to:", args.output)
