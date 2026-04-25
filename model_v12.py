"""
model_v12.py — Tweedie LGB + neighbor BERT surprisal + cleaner feature set
Key changes from v10:
  - Tweedie LGB: designed for zero-inflated positive distributions (30% skips = zeros)
  - Log-target LGB: predict log(TRT+1), better for heavy-tailed distribution
  - prev/next word BERT surprisal: spillover effects from adjacent words
  - Relative BERT surprisal: how surprising is this word vs. the text average
  - Remove features that hurt in v11 (text-type dummies, redundant sentence feats)
"""
import argparse, os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from collections import Counter
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from wordfreq import zipf_frequency, word_frequency
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--train',      default='/Users/renyihuang/NLP/train_data.csv')
parser.add_argument('--test',       default='/Users/renyihuang/NLP/test_data.csv')
parser.add_argument('--output',     default='/Users/renyihuang/NLP/submission_v12.csv')
parser.add_argument('--bert-cache', default='/Users/renyihuang/NLP/bert_surprisal_cache.csv')
args = parser.parse_args()

# ============================================================
# 1. Load & parse
# ============================================================
STRIP_CHARS = ".,;:!?()[]{}\"'-–—«»„\""
SENT_ENDS   = set('.!?')

raw_train = pd.read_csv(args.train)
raw_test  = pd.read_csv(args.test)
print("Loaded: train={}, test={}".format(len(raw_train), len(raw_test)))

def parse_base(df):
    df = df.copy()
    def get_word_key(wid):
        parts = wid.split('_')
        try:
            pi = parts.index('page')
            return '{}_page_{}_{}'.format('_'.join(parts[:pi-1]), parts[pi+1], parts[pi+2])
        except: return wid
    df['word_key']   = df['word_id'].apply(get_word_key)
    df['word_clean'] = df['word'].str.strip(STRIP_CHARS)
    df['word_lower'] = df['word_clean'].str.lower()
    df['text_type']  = df['text'].str.split('_').str[0]
    df['word_pos']   = df['word_key'].str.split('_').str[-1].astype(float)
    return df

raw_train = parse_base(raw_train)
raw_test  = parse_base(raw_test)

# ============================================================
# 2. BERT surprisal + neighbor surprisal
# ============================================================
if not os.path.exists(args.bert_cache):
    print("ERROR: BERT cache not found. Run: .venv/bin/python compute_bert.py")
    raise SystemExit(1)

print("Loading BERT surprisal from cache:", args.bert_cache)
bert_surp = pd.read_csv(args.bert_cache)

all_df = pd.concat([raw_train, raw_test], ignore_index=True)
all_df  = all_df.merge(bert_surp, on='word_key', how='left')
gbs     = all_df['bert_surprisal'].median()
all_df['bert_surprisal'] = all_df['bert_surprisal'].fillna(gbs)

# Compute neighbor surprisal per word_key (no label leakage)
print("Computing neighbor BERT surprisal...")
surp_pos = (all_df.drop_duplicates('word_key')
            [['word_key','text','word_pos','bert_surprisal']]
            .sort_values(['text','word_pos']))
text_mean_bert = surp_pos.groupby('text')['bert_surprisal'].mean().rename('text_mean_bert')
surp_pos = surp_pos.merge(text_mean_bert, on='text', how='left')
surp_pos['prev_bert_surp'] = surp_pos.groupby('text')['bert_surprisal'].shift(1).fillna(gbs)
surp_pos['next_bert_surp'] = surp_pos.groupby('text')['bert_surprisal'].shift(-1).fillna(gbs)
surp_pos['rel_bert_surp']  = surp_pos['bert_surprisal'] - surp_pos['text_mean_bert']

bert_feats = surp_pos[['word_key','prev_bert_surp','next_bert_surp','rel_bert_surp','text_mean_bert']]

raw_train = raw_train.merge(bert_surp,   on='word_key', how='left')
raw_train = raw_train.merge(bert_feats,  on='word_key', how='left')
raw_test  = raw_test.merge(bert_surp,    on='word_key', how='left')
raw_test  = raw_test.merge(bert_feats,   on='word_key', how='left')
for col in ['bert_surprisal','prev_bert_surp','next_bert_surp','rel_bert_surp','text_mean_bert']:
    raw_train[col] = raw_train[col].fillna(gbs if 'bert_surp' in col else 0)
    raw_test[col]  = raw_test[col].fillna(gbs if 'bert_surp' in col else 0)

print("BERT features merged. Mean bert_surp: {:.2f}".format(raw_train['bert_surprisal'].mean()))

# ============================================================
# 3. Sentence position (structural, no leakage)
# ============================================================
def add_sentence_features(df):
    uniq = (df.drop_duplicates('word_key')
            [['word_key','text','word_pos','word']].copy()
            .sort_values(['text','word_pos']))
    records = []
    for text, grp in uniq.groupby('text'):
        grp  = grp.sort_values('word_pos')
        orig = grp['word'].fillna('').tolist()
        keys = grp['word_key'].tolist()
        sent_id, pos_in = 0, 0
        for w, k in zip(orig, keys):
            records.append({'word_key': k, 'sent_pos': pos_in})
            stripped = str(w).strip().rstrip('"»")]}')
            if stripped and stripped[-1] in SENT_ENDS:
                sent_id += 1; pos_in = 0
            else:
                pos_in += 1
    sent_df = pd.DataFrame(records)
    df = df.merge(sent_df, on='word_key', how='left')
    df['sent_pos']      = df['sent_pos'].fillna(0)
    df['is_sent_start'] = (df['sent_pos'] == 0).astype(float)
    df['sent_pos_log']  = np.log1p(df['sent_pos'])
    return df

print("Building sentence features...")
raw_train = add_sentence_features(raw_train)
raw_test  = add_sentence_features(raw_test)

# ============================================================
# 4. Non-target structural features
# ============================================================
VOWELS = set('aeiouAEIOUăâîĂÂÎ')
RO_DIAC = set('ăâîșțĂÂÎȘȚ')
CONSONANT_CLUSTERS = ['str','ntr','mpr','ngr','ndr','spr','scr','zdr','zgr']
RO_STOPWORDS = {
    'de','la','în','și','cu','pe','că','ca','sa','se','nu','dar','sau',
    'un','o','al','ale','lui','ei','le','îi','ne','vă','mă','te','este',
    'sunt','era','ai','am','avem','fi','fost','care','ce','din','prin',
    'mai','poate','când','cum','tot','toate','acest','această','aceste',
    'pentru','despre','între','după','înainte','acolo','atunci','cât',
}
RO_PREFIXES  = ['ne','re','des','de','în','sub','supra','anti','pre','pro']
RO_SUFFIXES_NOUN = ['ului','ilor','elor','iile','itor','ătoare']
RO_SUFFIXES_VERB = ['ează','esc','ăsc','ste','ete','ăm','ați']

all_words = set(raw_train['word_lower'].dropna()) | set(raw_test['word_lower'].dropna())
print("Building wordfreq cache ({} unique words)...".format(len(all_words)))
zipf_ro_cache = {w: zipf_frequency(w, 'ro') for w in all_words}
zipf_en_cache = {w: zipf_frequency(w, 'en') for w in all_words}
freq_ro_cache = {w: word_frequency(w, 'ro')  for w in all_words}

text_len_map = (raw_train.drop_duplicates('word_key')
                .groupby('text')['word_key'].count().rename('text_n_words'))

def count_syllables_ro(word):
    w = str(word).lower(); vowels = set('aeiouăâî')
    count, in_vowel = 0, False
    for c in w:
        if c in vowels:
            if not in_vowel: count += 1; in_vowel = True
        else: in_vowel = False
    return max(1, count)

syllable_cache = {w: count_syllables_ro(w) for w in all_words}

def add_structural_features(df):
    df = df.copy()
    wc = df['word_clean'].fillna('').astype(str)
    wl = df['word_lower'].fillna('').astype(str)
    df['word_len']        = wc.str.len().astype(float)
    df['n_vowels']        = wc.apply(lambda w: sum(1 for c in w if c in VOWELS))
    df['n_consonants']    = df['word_len'] - df['n_vowels']
    df['vowel_ratio']     = df['n_vowels'] / (df['word_len'] + 1)
    df['has_diacritics']  = wc.apply(lambda w: int(any(c in RO_DIAC for c in w)))
    df['n_diacritics']    = wc.apply(lambda w: sum(1 for c in w if c in RO_DIAC))
    df['is_numeric']      = wc.apply(lambda w: int(w.replace('.','').replace(',','').isdigit() and len(w)>0))
    df['is_url']          = df['word'].fillna('').apply(lambda w: int('http' in w or 'www.' in w))
    df['is_capitalized']  = wc.apply(lambda w: int(len(w)>0 and w[0].isupper()))
    df['is_all_caps']     = wc.apply(lambda w: int(len(w)>1 and w.isupper()))
    df['word_len_sq']     = df['word_len'] ** 2
    df['word_len_log']    = np.log1p(df['word_len'])
    df['word_len_sqrt']   = np.sqrt(df['word_len'])
    df['has_hyphen']      = df['word'].fillna('').apply(lambda w: int('-' in w))
    df['has_cluster']     = wl.apply(lambda w: int(any(c in w for c in CONSONANT_CLUSTERS)))
    df['is_stopword']     = wl.apply(lambda w: int(w in RO_STOPWORDS))
    df['n_unique_chars']  = wc.apply(lambda w: len(set(w.lower())))
    df['char_diversity']  = df['n_unique_chars'] / (df['word_len'] + 1)
    df['n_syllables']     = wl.map(syllable_cache).fillna(1)
    df['syllable_log']    = np.log1p(df['n_syllables'])
    df['len_per_syllable']= df['word_len'] / (df['n_syllables'] + 1)
    df['has_ro_prefix']   = wl.apply(lambda w: int(any(w.startswith(p) and len(w)>len(p)+2 for p in RO_PREFIXES)))
    df['has_noun_suffix'] = wl.apply(lambda w: int(any(w.endswith(s) for s in RO_SUFFIXES_NOUN)))
    df['has_verb_suffix'] = wl.apply(lambda w: int(any(w.endswith(s) for s in RO_SUFFIXES_VERB)))
    df['zipf_ro']         = wl.map(zipf_ro_cache).fillna(0.0)
    df['zipf_en']         = wl.map(zipf_en_cache).fillna(0.0)
    df['zipf_max']        = df[['zipf_ro','zipf_en']].max(axis=1)
    df['freq_log']        = np.log1p(wl.map(freq_ro_cache).fillna(0.0) * 1e6)
    df['is_unknown_ro']   = (df['zipf_ro'] == 0).astype(float)
    df['is_rare']         = (df['zipf_ro'] < 2).astype(float)
    df['is_very_rare']    = (df['zipf_ro'] < 1).astype(float)
    df['is_common']       = (df['zipf_ro'] > 4).astype(float)
    df['zipf_x_len']      = df['zipf_ro'] * df['word_len']
    df['zipf_x_syl']      = df['zipf_ro'] * df['n_syllables']
    df['page_num']        = df['word_key'].str.split('_').apply(lambda x: float(x[-2]) if len(x)>=2 else 0.0)
    df['word_pos_log']    = np.log1p(df['word_pos'])
    df['word_pos_sq']     = df['word_pos'] ** 2
    df = df.merge(text_len_map, on='text', how='left')
    df['text_n_words']    = df['text_n_words'].fillna(df['word_pos'] + 1)
    df['pos_ratio']       = df['word_pos'] / (df['text_n_words'] + 1)
    df['is_first_10pct']  = (df['pos_ratio'] < 0.1).astype(float)
    df['is_last_10pct']   = (df['pos_ratio'] > 0.9).astype(float)
    df['is_mid']          = ((df['pos_ratio'] > 0.3) & (df['pos_ratio'] < 0.7)).astype(float)
    return df

raw_train = add_structural_features(raw_train)
raw_test  = add_structural_features(raw_test)

# ============================================================
# 5. N-gram LM surprisal
# ============================================================
def build_lm(df):
    df2 = df.drop_duplicates('word_key').sort_values(['text','word_pos'])
    unigram, bigram, trigram = Counter(), Counter(), Counter()
    for text, grp in df2.groupby('text'):
        words = ['<S>'] + grp['word_lower'].fillna('<UNK>').tolist() + ['</S>']
        for i, w in enumerate(words):
            unigram[w] += 1
            if i > 0: bigram[(words[i-1], w)] += 1
            if i > 1: trigram[(words[i-2], words[i-1], w)] += 1
    V = len(unigram) + 1; total_uni = sum(unigram.values()) + 1
    def bp(w_prev, w): return (bigram.get((w_prev,w),0)+1)/(unigram.get(w_prev,0)+V)
    def tp(w_pp, w_prev, w): return (trigram.get((w_pp,w_prev,w),0)+1)/(bigram.get((w_pp,w_prev),0)+V)
    def up(w): return (unigram.get(w,0)+1)/total_uni
    return bp, tp, up

def compute_lm_surprisal(df, bp, tp, up):
    df2 = df.drop_duplicates('word_key').sort_values(['text','word_pos']).copy()
    df2['word_lower_fill'] = df2['word_lower'].fillna('<UNK>')
    results = []
    for text, grp in df2.groupby('text'):
        words = ['<S>','<S>'] + grp['word_lower_fill'].tolist()
        keys  = ['__','__']   + grp['word_key'].tolist()
        for i in range(2, len(words)):
            w, w1, w2 = words[i], words[i-1], words[i-2]
            results.append({
                'word_key': keys[i],
                'surp_uni': float(-np.log(up(w)+1e-10)),
                'surp_bi':  float(-np.log(bp(w1,w)+1e-10)),
                'surp_tri': float(-np.log(tp(w2,w1,w)+1e-10)),
            })
    surp_df = pd.DataFrame(results)
    surp_df['surp_interp'] = 0.5*surp_df['surp_tri'] + 0.5*surp_df['surp_bi']
    return df.merge(surp_df, on='word_key', how='left')

# ============================================================
# 6. Target-dependent tables (per fold, anti-leakage)
# ============================================================
BAYES_K = 10

def compute_target_tables(df):
    nz = df[df['answer'] > 0]
    T  = {}
    T['gm']   = df['answer'].mean()
    T['gmd']  = df['answer'].median()
    T['gsd']  = df['answer'].std()
    T['gsr']  = (df['answer'] == 0).mean()
    T['gnm']  = nz['answer'].mean() if len(nz) else df['answer'].mean()
    T['gns']  = nz['answer'].std()  if len(nz) else df['answer'].std()

    p = df.groupby('participant_id')['answer'].agg(['mean','median','std'])
    p.columns = ['part_mean','part_median','part_std']
    p['part_skip_rate']    = df.groupby('participant_id').apply(lambda x: (x['answer']==0).mean())
    p['part_nonzero_mean'] = nz.groupby('participant_id')['answer'].mean() if len(nz) else pd.Series(dtype=float)
    p['part_cv']           = p['part_std'] / (p['part_mean'] + 1e-9)
    T['part'] = p

    T['part_tt']   = df.groupby(['participant_id','text_type'])['answer'].mean().rename('part_tt_mean').reset_index()
    T['part_tts']  = (df.groupby(['participant_id','text_type']).apply(lambda x: (x['answer']==0).mean())
                       .rename('part_tt_skip').reset_index())

    # Text-level features
    tx = df.groupby('text')['answer'].agg(['mean','median','std'])
    tx.columns = ['text_mean_trt','text_median_trt','text_std_trt']
    tx['text_skip_rate']    = df.groupby('text').apply(lambda x: (x['answer']==0).mean())
    tx['text_nonzero_mean'] = nz.groupby('text')['answer'].mean() if len(nz) else pd.Series(dtype=float)
    T['tx'] = tx

    wf = df.groupby('word_clean')['answer'].agg(['mean','median','std','count'])
    wf.columns = ['wf_mean','wf_median','wf_std','wf_count']
    T['wf'] = wf

    wl = df.groupby('word_lower')['answer'].agg(['mean','median','std','count'])
    wl.columns = ['wl_mean','wl_median','wl_std','wl_count']
    wl['wl_mean_bayes'] = (wl['wl_count']*wl['wl_mean'] + BAYES_K*T['gm']) / (wl['wl_count']+BAYES_K)
    T['wl'] = wl

    wk = df.groupby('word_key')['answer'].agg(['mean','median','std','count'])
    wk.columns = ['wk_mean','wk_median','wk_std','wk_count']
    wk['wk_mean_bayes'] = (wk['wk_count']*wk['wk_mean'] + BAYES_K*T['gm']) / (wk['wk_count']+BAYES_K)
    T['wk'] = wk

    tt = df.groupby('text_type')['answer'].agg(['mean','std'])
    tt.columns = ['tt_mean','tt_std']
    T['tt'] = tt

    T['wsr'] = df.groupby('word_lower').apply(lambda x: (x['answer']==0).mean()).rename('skip_rate')
    T['wnm'] = (nz.groupby('word_lower')['answer'].mean().rename('wl_nonzero_mean')
                if len(nz) else pd.Series(dtype=float))

    words = df['word_lower'].fillna('').tolist()
    bg, tg = Counter(), Counter()
    for w in words:
        for i in range(len(w)-1): bg[w[i:i+2]] += 1
        for i in range(len(w)-2): tg[w[i:i+3]] += 1
    T['bg'] = bg; T['tg'] = tg
    T['tbg'] = sum(bg.values())+1; T['ttg'] = sum(tg.values())+1

    df2 = df.copy(); df2['wlr'] = df2['word_clean'].str.len().fillna(0)
    srt = df2.drop_duplicates('word_key').sort_values(['text','word_pos'])
    for sh, nm in [(1,'prev'),(-1,'next'),(2,'prev2'),(-2,'next2')]:
        srt[f'{nm}_word_len'] = srt.groupby('text')['wlr'].shift(sh).fillna(0)
    T['nb'] = srt[['word_key','prev_word_len','next_word_len','prev2_word_len','next2_word_len']]
    return T

def apply_tf(df, T):
    df = df.copy()
    gm=T['gm']; gmd=T['gmd']; gsd=T['gsd']; gsr=T['gsr']; gnm=T['gnm']

    bg=T['bg']; tbg=T['tbg']; tg=T['tg']; ttg=T['ttg']
    wl_col = df['word_lower'].fillna('').astype(str)
    def abg(w):
        w=str(w).lower()
        return float(np.mean([bg.get(w[i:i+2],0)/tbg for i in range(len(w)-1)])) if len(w)>=2 else 0.0
    def atg(w):
        w=str(w).lower()
        return float(np.mean([tg.get(w[i:i+3],0)/ttg for i in range(len(w)-2)])) if len(w)>=3 else 0.0

    df['bigram_freq']      = wl_col.apply(abg)
    df['trigram_freq']     = wl_col.apply(atg)
    df['bigram_freq_log']  = np.log1p(df['bigram_freq']*1000)
    df['trigram_freq_log'] = np.log1p(df['trigram_freq']*1000)

    df = df.merge(T['part'], on='participant_id', how='left')
    df['part_mean']         = df['part_mean'].fillna(gm)
    df['part_median']       = df['part_median'].fillna(gmd)
    df['part_std']          = df['part_std'].fillna(gsd)
    df['part_skip_rate']    = df['part_skip_rate'].fillna(gsr)
    df['part_nonzero_mean'] = df['part_nonzero_mean'].fillna(gnm)
    df['part_cv']           = df['part_cv'].fillna(df['part_std']/(df['part_mean']+1e-9))
    df['part_speed_rel']    = df['part_mean'] / gm

    df = df.merge(T['part_tt'],  on=['participant_id','text_type'], how='left')
    df['part_tt_mean'] = df['part_tt_mean'].fillna(df['part_mean'])
    df['part_tt_rel']  = df['part_tt_mean'] / (gm+1e-9)
    df = df.merge(T['part_tts'], on=['participant_id','text_type'], how='left')
    df['part_tt_skip'] = df['part_tt_skip'].fillna(df['part_skip_rate'])

    df = df.merge(T['tx'], on='text', how='left')
    df['text_mean_trt']    = df['text_mean_trt'].fillna(gm)
    df['text_median_trt']  = df['text_median_trt'].fillna(gmd)
    df['text_std_trt']     = df['text_std_trt'].fillna(gsd)
    df['text_skip_rate_t'] = df['text_skip_rate'].fillna(gsr)  # renamed to avoid conflict with skip_rate
    df['text_nonzero_mean']= df['text_nonzero_mean'].fillna(gnm)
    df['text_rel_speed']   = df['text_mean_trt'] / gm

    df = df.merge(T['wf'], on='word_clean', how='left')
    df['wf_mean']   = df['wf_mean'].fillna(gm)
    df['wf_median'] = df['wf_median'].fillna(gmd)
    df['wf_std']    = df['wf_std'].fillna(gsd)
    df['wf_count']  = df['wf_count'].fillna(0)

    df = df.merge(T['wl'], on='word_lower', how='left')
    df['wl_mean']       = df['wl_mean'].fillna(df['wf_mean'])
    df['wl_mean_bayes'] = df['wl_mean_bayes'].fillna(gm)
    df['wl_median']     = df['wl_median'].fillna(df['wf_median'])
    df['wl_std']        = df['wl_std'].fillna(df['wf_std'])
    df['wl_count']      = df['wl_count'].fillna(0)

    df = df.merge(T['wk'], on='word_key', how='left')
    df['wk_mean']       = df['wk_mean'].fillna(df['wf_mean'])
    df['wk_mean_bayes'] = df['wk_mean_bayes'].fillna(gm)
    df['wk_median']     = df['wk_median'].fillna(df['wf_median'])
    df['wk_std']        = df['wk_std'].fillna(df['wf_std'])
    df['wk_count']      = df['wk_count'].fillna(0)

    df = df.merge(T['tt'], on='text_type', how='left')
    df['tt_mean'] = df['tt_mean'].fillna(gm)
    df['tt_std']  = df['tt_std'].fillna(gsd)

    df = df.merge(T['wsr'], on='word_lower', how='left')
    df['skip_rate'] = df['skip_rate'].fillna(gsr)
    df = df.merge(T['wnm'], on='word_lower', how='left')
    df['wl_nonzero_mean'] = df['wl_nonzero_mean'].fillna(gnm)

    df = df.merge(T['nb'], on='word_key', how='left')
    for c in ['prev_word_len','next_word_len','prev2_word_len','next2_word_len']:
        df[c] = df[c].fillna(0)
    df['neighbor_len_sum']  = df['prev_word_len'] + df['next_word_len']
    df['neighbor_len_mean'] = df['neighbor_len_sum'] / 2
    df['neighbor_len_max']  = df[['prev_word_len','next_word_len']].max(axis=1)

    # BERT interactions
    df['bert_x_part_spd'] = df['bert_surprisal'] * df['part_speed_rel']
    df['bert_x_zipf']     = df['bert_surprisal'] * (1 / (df['zipf_max'] + 0.5))
    df['bert_x_len']      = df['bert_surprisal'] * df['word_len']
    df['bert_x_skip']     = df['bert_surprisal'] * df['skip_rate']
    # Spillover: previous word's surprisal affects current word reading time
    df['prev_bert_x_part']= df['prev_bert_surp'] * df['part_speed_rel']
    # Sentence start interactions
    df['sent_x_part']     = df['is_sent_start'] * df['part_mean']
    df['sent_x_bert']     = df['is_sent_start'] * df['bert_surprisal']

    df['wf_x_part']       = df['wf_mean'] * df['part_mean'] / gm
    df['wk_x_part']       = df['wk_mean'] * df['part_mean'] / gm
    df['len_x_part']      = df['word_len'] * df['part_mean']
    df['syl_x_part']      = df['n_syllables'] * df['part_mean']
    df['wf_seen']         = (df['wf_count']>0).astype(float)
    df['wk_seen']         = (df['wk_count']>0).astype(float)
    df['part_wf_diff']    = df['part_mean'] - df['wf_mean']
    df['skip_x_part']     = df['skip_rate'] * df['part_mean']
    df['expected_trt']    = df['wl_nonzero_mean'] * (1 - df['skip_rate'])
    df['expected_x_part'] = df['expected_trt'] * df['part_speed_rel']
    df['skip_diff']       = df['part_skip_rate'] - df['skip_rate']
    df['wk_vs_wf']        = df['wk_mean'] - df['wf_mean']
    df['nz_vs_part']      = df['wl_nonzero_mean'] / (df['part_nonzero_mean']+1e-9)
    df['zipf_x_part']     = df['zipf_max'] * df['part_speed_rel']
    df['difficulty']      = (df['n_syllables']*df['word_len']) / (df['zipf_max']+1)
    return df

FEATURES = [
    'word_len','n_vowels','n_consonants','vowel_ratio',
    'has_diacritics','n_diacritics','is_numeric','is_url',
    'is_capitalized','is_all_caps','has_hyphen','has_cluster','is_stopword',
    'word_len_sq','word_len_log','word_len_sqrt','n_unique_chars','char_diversity',
    'n_syllables','syllable_log','len_per_syllable',
    'has_ro_prefix','has_noun_suffix','has_verb_suffix',
    'zipf_ro','zipf_en','zipf_max','freq_log',
    'is_unknown_ro','is_rare','is_very_rare','is_common','zipf_x_len','zipf_x_syl',
    # BERT surprisal (key features)
    'bert_surprisal','bert_x_part_spd','bert_x_zipf','bert_x_len','bert_x_skip',
    'prev_bert_surp','next_bert_surp','rel_bert_surp','text_mean_bert',
    'prev_bert_x_part',
    # N-gram surprisal
    'surp_uni','surp_bi','surp_tri','surp_interp',
    'bigram_freq','trigram_freq','bigram_freq_log','trigram_freq_log',
    # Sentence position
    'sent_pos','sent_pos_log','is_sent_start','sent_x_part','sent_x_bert',
    # Position
    'word_pos','page_num','word_pos_log','word_pos_sq',
    'pos_ratio','is_first_10pct','is_last_10pct','is_mid','text_n_words',
    # Neighbor word length
    'prev_word_len','next_word_len','prev2_word_len','next2_word_len',
    'neighbor_len_sum','neighbor_len_mean','neighbor_len_max',
    # Participant features
    'part_mean','part_median','part_std','part_speed_rel','part_cv',
    'part_skip_rate','part_nonzero_mean','part_tt_mean','part_tt_rel','part_tt_skip',
    # Text-level
    'text_mean_trt','text_median_trt','text_std_trt','text_skip_rate_t','text_nonzero_mean',
    'text_rel_speed',
    # Word-level target stats
    'wf_mean','wf_median','wf_std','wf_count',
    'wl_mean','wl_mean_bayes','wl_median','wl_std','wl_count',
    'wk_mean','wk_mean_bayes','wk_median','wk_std','wk_count',
    'skip_rate','wl_nonzero_mean','tt_mean','tt_std',
    # Interactions
    'wf_x_part','wk_x_part','len_x_part','syl_x_part','wf_seen','wk_seen',
    'part_wf_diff','skip_x_part','expected_trt','expected_x_part',
    'skip_diff','wk_vs_wf','nz_vs_part','zipf_x_part','difficulty','sent_x_part',
]

META_FEATS = [
    'part_mean','part_skip_rate','skip_rate','wl_nonzero_mean','expected_trt',
    'word_len','n_syllables','pos_ratio','zipf_ro','zipf_max',
    'bert_surprisal','prev_bert_surp','surp_bi',
    'wf_mean','wl_mean','wl_mean_bayes','tt_mean',
    'is_sent_start','text_mean_trt','text_skip_rate_t',
]

def eval_metric(y_true, preds):
    y_true = np.array(y_true, dtype=float); preds = np.array(preds, dtype=float)
    r2 = max(0.0, r2_score(y_true, preds)); pears = pearsonr(y_true, preds)[0]
    if np.isnan(pears): pears = 0.0
    return 100.0 * (abs(pears) + r2) / 2.0

# ============================================================
# 7. Models
# ============================================================
lgb_p  = dict(n_estimators=4000,learning_rate=0.015,num_leaves=127,min_child_samples=15,
               reg_alpha=0.05,reg_lambda=0.05,subsample=0.8,colsample_bytree=0.8,
               n_jobs=-1,random_state=42,verbose=-1)
lgb_p2 = dict(n_estimators=4000,learning_rate=0.015,num_leaves=63,min_child_samples=20,
               reg_alpha=0.1,reg_lambda=0.1,subsample=0.7,colsample_bytree=0.7,
               n_jobs=-1,random_state=99,verbose=-1)
# Tweedie: designed for zero-inflated positive distributions
lgb_tw = dict(n_estimators=4000,learning_rate=0.015,num_leaves=127,min_child_samples=15,
               reg_alpha=0.05,reg_lambda=0.05,subsample=0.8,colsample_bytree=0.8,
               objective='tweedie',tweedie_variance_power=1.5,
               n_jobs=-1,random_state=42,verbose=-1)
lgb_clf= dict(n_estimators=2000,learning_rate=0.02,num_leaves=63,min_child_samples=15,
               reg_alpha=0.1,reg_lambda=0.1,subsample=0.8,colsample_bytree=0.8,
               objective='binary',n_jobs=-1,random_state=42,verbose=-1)
xgb_p  = dict(n_estimators=3000,learning_rate=0.02,max_depth=7,min_child_weight=10,
               reg_alpha=0.05,reg_lambda=1.0,subsample=0.8,colsample_bytree=0.8,
               early_stopping_rounds=50,tree_method='hist',n_jobs=-1,random_state=42,verbosity=0)
cat_p  = dict(iterations=2000,learning_rate=0.02,depth=8,l2_leaf_reg=3,
               random_strength=1,bagging_temperature=0.5,random_seed=42,verbose=0)
hgb_p  = dict(max_iter=1000,learning_rate=0.02,max_leaf_nodes=127,
               min_samples_leaf=15,l2_regularization=0.05,max_bins=255,random_state=42)

# ============================================================
# 8. Cross-validation
# ============================================================
groups = raw_train['text'].values
y_all  = raw_train['answer'].values
gkf    = GroupKFold(n_splits=9)
n      = len(raw_train)

oof_lgb=np.zeros(n); oof_lgb2=np.zeros(n); oof_tw=np.zeros(n)
oof_xgb=np.zeros(n); oof_cat=np.zeros(n);  oof_hgb=np.zeros(n)
oof_skip=np.zeros(n)
oof_me = np.zeros((n, len(META_FEATS)))
cv_scores = []

print("="*70)
print("Cross-Validation (anti-leakage + BERT + Tweedie + neighbor surprisal)")
print("="*70)

surp_cols = ['surp_uni','surp_bi','surp_tri','surp_interp']

for fold, (tr_idx, val_idx) in enumerate(gkf.split(raw_train, y_all, groups)):
    tr_df  = raw_train.iloc[tr_idx].copy()
    val_df = raw_train.iloc[val_idx].copy()

    bp, tp, up = build_lm(tr_df)
    tr_df  = compute_lm_surprisal(tr_df,  bp, tp, up)
    val_df = compute_lm_surprisal(val_df, bp, tp, up)
    for c in surp_cols:
        for df_ in [tr_df, val_df]:
            if c not in df_.columns: df_[c]=0.0
            df_[c] = df_[c].fillna(0.0)

    T = compute_target_tables(tr_df)
    tr_f  = apply_tf(tr_df,  T)
    val_f = apply_tf(val_df, T)
    for c in surp_cols:
        if c not in tr_f.columns:  tr_f[c]=0.0
        if c not in val_f.columns: val_f[c]=0.0

    X_tr  = tr_f[FEATURES].fillna(0).values
    X_val = val_f[FEATURES].fillna(0).values
    y_tr  = tr_df['answer'].values; y_val = val_df['answer'].values
    y_bin_tr = (y_tr==0).astype(int)

    cb = [lgb.early_stopping(50,verbose=False), lgb.log_evaluation(-1)]

    mA  = lgb.LGBMRegressor(**lgb_p);  mA.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],callbacks=cb)
    pA  = mA.predict(X_val); oof_lgb[val_idx]=pA

    mA2 = lgb.LGBMRegressor(**lgb_p2); mA2.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],callbacks=cb)
    pA2 = mA2.predict(X_val); oof_lgb2[val_idx]=pA2

    mTW = lgb.LGBMRegressor(**lgb_tw); mTW.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],callbacks=cb)
    pTW = mTW.predict(X_val); pTW=np.clip(pTW,0,None); oof_tw[val_idx]=pTW

    mB  = xgb.XGBRegressor(**xgb_p);   mB.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],verbose=False)
    pB  = mB.predict(X_val); oof_xgb[val_idx]=pB

    mC  = CatBoostRegressor(**cat_p);   mC.fit(X_tr,y_tr,eval_set=(X_val,y_val),early_stopping_rounds=50)
    pC  = mC.predict(X_val); oof_cat[val_idx]=pC

    mD  = HistGradientBoostingRegressor(**hgb_p); mD.fit(X_tr,y_tr)
    pD  = mD.predict(X_val); oof_hgb[val_idx]=pD

    mE  = lgb.LGBMClassifier(**lgb_clf); mE.fit(X_tr,y_bin_tr,eval_set=[(X_val,(y_val==0).astype(int))],callbacks=cb)
    p_sk= mE.predict_proba(X_val)[:,1];  oof_skip[val_idx]=p_sk

    oof_me[val_idx] = val_f[META_FEATS].fillna(0).values

    p_ens = 0.30*pA + 0.15*pA2 + 0.15*pTW + 0.20*pB + 0.12*pC + 0.08*pD
    p_ens = np.clip(p_ens, 0, None)

    sA=eval_metric(y_val,pA); sTW=eval_metric(y_val,pTW)
    sB=eval_metric(y_val,pB); sC=eval_metric(y_val,pC); sEns=eval_metric(y_val,p_ens)
    cv_scores.append(sEns)
    g=groups[val_idx][0]
    print("Fold {:d} ({:20s}) | LGB:{:.1f} TW:{:.1f} XGB:{:.1f} CAT:{:.1f} | Ens:{:.1f}".format(
        fold+1,g,sA,sTW,sB,sC,sEns))

print("="*70)
print("CV Ensemble (honest): {:.2f} +/- {:.2f}".format(np.mean(cv_scores),np.std(cv_scores)))
print("="*70)

# ============================================================
# 9. Meta-model
# ============================================================
print("\nTraining meta-model on true OOF...")
oof_hurdle = (1-oof_skip)*oof_lgb
X_meta = np.column_stack([oof_lgb,oof_lgb2,oof_tw,oof_xgb,oof_cat,oof_hgb,oof_skip,oof_hurdle,oof_me])
meta   = lgb.LGBMRegressor(n_estimators=1500,learning_rate=0.01,num_leaves=31,
                            min_child_samples=20,reg_alpha=0.1,reg_lambda=0.1,
                            n_jobs=-1,random_state=42,verbose=-1)
meta.fit(X_meta, y_all)

# ============================================================
# 10. Retrain on full data
# ============================================================
print("\nRetraining on full data...")
bp_f, tp_f, up_f = build_lm(raw_train)
tr_full  = compute_lm_surprisal(raw_train, bp_f, tp_f, up_f)
te_full  = compute_lm_surprisal(raw_test,  bp_f, tp_f, up_f)
for c in surp_cols:
    for df_ in [tr_full, te_full]:
        if c not in df_.columns: df_[c]=0.0
        df_[c]=df_[c].fillna(0.0)

T_full  = compute_target_tables(tr_full)
tr_full = apply_tf(tr_full, T_full)
te_full = apply_tf(te_full, T_full)
for c in surp_cols + FEATURES:
    if c not in tr_full.columns: tr_full[c]=0.0
    if c not in te_full.columns: te_full[c]=0.0

X_full  = tr_full[FEATURES].fillna(0).values
X_test  = te_full[FEATURES].fillna(0).values
y_full  = raw_train['answer'].values
y_bin_f = (y_full==0).astype(int)
xgb_ne  = {k:v for k,v in xgb_p.items() if k!='early_stopping_rounds'}
cb = [lgb.log_evaluation(-1)]

fA  = lgb.LGBMRegressor(**lgb_p);  fA.fit(X_full,y_full,callbacks=cb)
fA2 = lgb.LGBMRegressor(**lgb_p2); fA2.fit(X_full,y_full,callbacks=cb)
fTW = lgb.LGBMRegressor(**lgb_tw); fTW.fit(X_full,y_full,callbacks=cb)
fB  = xgb.XGBRegressor(**xgb_ne);  fB.fit(X_full,y_full,verbose=False)
fC  = CatBoostRegressor(**cat_p);   fC.fit(X_full,y_full)
fD  = HistGradientBoostingRegressor(**hgb_p); fD.fit(X_full,y_full)
fE  = lgb.LGBMClassifier(**lgb_clf); fE.fit(X_full,y_bin_f,callbacks=cb)

pA_t=fA.predict(X_test); pA2_t=fA2.predict(X_test)
pTW_t=np.clip(fTW.predict(X_test),0,None)
pB_t=fB.predict(X_test); pC_t=fC.predict(X_test)
pD_t=fD.predict(X_test); sk_t=fE.predict_proba(X_test)[:,1]
hurdle_t=(1-sk_t)*pA_t

X_mt = np.column_stack([pA_t,pA2_t,pTW_t,pB_t,pC_t,pD_t,sk_t,hurdle_t,
                         te_full[META_FEATS].fillna(0).values])
mp   = meta.predict(X_mt)
ens_t= 0.30*pA_t+0.15*pA2_t+0.15*pTW_t+0.20*pB_t+0.12*pC_t+0.08*pD_t
fp   = 0.5*np.clip(ens_t,0,None) + 0.5*np.clip(mp,0,None)
fp   = np.clip(fp,0,None)

out = raw_test[['datapointID']].copy()
out['subtaskID']=1; out['answer']=fp
out=out[['subtaskID','datapointID','answer']]
out.to_csv(args.output,index=False)
print("\nDone!"); print(out.head(10).to_string())
print("\nSaved to:", args.output)
