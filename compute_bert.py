"""
compute_bert.py — standalone BERT surprisal computation
Run this FIRST before model_v10.py to build the cache.
No LightGBM/CatBoost imports so no OpenMP conflict with PyTorch.
"""
import os, argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

parser = argparse.ArgumentParser()
parser.add_argument('--train',      default='/Users/renyihuang/NLP/train_data.csv')
parser.add_argument('--test',       default='/Users/renyihuang/NLP/test_data.csv')
parser.add_argument('--cache',      default='/Users/renyihuang/NLP/bert_surprisal_cache.csv')
parser.add_argument('--model',      default='bert-base-multilingual-cased')
parser.add_argument('--batch-size', type=int, default=8)
args = parser.parse_args()

if os.path.exists(args.cache):
    print("Cache already exists:", args.cache)
    df = pd.read_csv(args.cache)
    print(f"  {len(df)} entries. Done.")
    exit(0)

# ── Load data ──────────────────────────────────────────────────
STRIP = ".,;:!?()[]{}\"'-–—«»„\""
def parse(df):
    df = df.copy()
    def wk(wid):
        p = wid.split('_')
        try:
            pi = p.index('page')
            return '{}_page_{}_{}'.format('_'.join(p[:pi-1]), p[pi+1], p[pi+2])
        except: return wid
    df['word_key']   = df['word_id'].apply(wk)
    df['word_lower'] = df['word'].str.strip(STRIP).str.lower()
    df['text_type']  = df['text'].str.split('_').str[0]
    df['word_pos']   = df['word_key'].str.split('_').str[-1].astype(float)
    return df

print("Loading data...")
train = parse(pd.read_csv(args.train))
test  = parse(pd.read_csv(args.test))
all_df = pd.concat([train, test], ignore_index=True)

# ── BERT setup ─────────────────────────────────────────────────
device = 'cpu'  # MPS crashes with BERT MLM; CPU is safe
print(f"BERT device: {device}")
print(f"Loading {args.model}...")
tokenizer = AutoTokenizer.from_pretrained(args.model)
model     = AutoModelForMaskedLM.from_pretrained(args.model).to(device)
model.eval()

# ── Unique word positions ──────────────────────────────────────
uniq = (all_df.drop_duplicates('word_key')
        [['word_key','word_lower','text','word_pos']].copy()
        .sort_values(['text','word_pos']))

text_words = {}
text_keys  = {}
for text, grp in uniq.groupby('text'):
    grp = grp.sort_values('word_pos')
    text_words[text] = grp['word_lower'].fillna('<unk>').tolist()
    text_keys[text]  = grp['word_key'].tolist()

total = sum(len(v) for v in text_words.values())
print(f"Computing surprisal for {total} unique positions (batch={args.batch_size})...")

results = {}
done = 0

for text, words in text_words.items():
    keys = text_keys[text]
    n    = len(words)

    for batch_start in range(0, n, args.batch_size):
        batch_end = min(batch_start + args.batch_size, n)
        input_ids_list   = []
        target_positions = []
        target_token_ids = []
        valid_indices    = []  # which positions in batch were successfully encoded

        for i in range(batch_start, batch_end):
            ctx_s = max(0, i - 60)
            ctx_e = min(n, i + 60)
            ctx   = words[ctx_s:ctx_e]
            li    = i - ctx_s

            original = ctx[li]
            masked   = ctx[:li] + [tokenizer.mask_token] + ctx[li+1:]
            text_str = ' '.join(masked)

            enc = tokenizer(text_str, return_tensors='pt',
                            truncation=True, max_length=512, padding=False)
            ids = enc['input_ids'][0]

            mask_pos_list = (ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            if len(mask_pos_list) == 0:
                results[keys[i]] = 10.0
                continue

            mask_pos  = mask_pos_list[0].item()
            orig_ids  = tokenizer(original, add_special_tokens=False)['input_ids']
            target_id = orig_ids[0] if orig_ids else tokenizer.unk_token_id

            input_ids_list.append(ids.tolist())
            target_positions.append(mask_pos)
            target_token_ids.append(target_id)
            valid_indices.append(i)

        if not input_ids_list:
            done += batch_end - batch_start
            continue

        max_len  = max(len(x) for x in input_ids_list)
        padded   = [x + [tokenizer.pad_token_id]*(max_len-len(x)) for x in input_ids_list]
        att_mask = [[1]*len(x) + [0]*(max_len-len(x)) for x in input_ids_list]

        inp = torch.tensor(padded,   device=device)
        att = torch.tensor(att_mask, device=device)

        with torch.no_grad():
            logits = model(input_ids=inp, attention_mask=att).logits

        for j, (pos, tok_id, orig_i) in enumerate(
                zip(target_positions, target_token_ids, valid_indices)):
            lp  = torch.log_softmax(logits[j, pos], dim=-1)
            s   = -lp[tok_id].item()
            results[keys[orig_i]] = float(np.clip(s, 0, 30))

        done += batch_end - batch_start
        if done % 200 == 0 or done == total:
            print(f"  {done}/{total} done ({100*done/total:.1f}%)")

# ── Save cache ─────────────────────────────────────────────────
surp_df = pd.DataFrame({'word_key':       list(results.keys()),
                        'bert_surprisal': list(results.values())})
surp_df.to_csv(args.cache, index=False)
print(f"\nSaved {len(surp_df)} entries to {args.cache}")
print("Now run: .venv/bin/python model_v10.py")
