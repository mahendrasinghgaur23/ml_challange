"""
Stage 2: Multi-Key Blocking (Memory-Safe for Google Colab / Limited RAM)
=============================================================
Root cause of exit 137: The previous sparse dot-product approach materializes
a (n_s1 × n_candidates) dense intermediate matrix for EACH chunk — on 3M US
candidates that's 500 × 3_000_000 × 4 bytes = ~6 GB per chunk call, blowing
Colab's 12-13 GB RAM limit even with aggressive GC.

Fix: Use sklearn.NearestNeighbors(metric='cosine') which computes top-K
internally WITHOUT ever materializing the full similarity matrix. It returns
only (n_s1, K) index/distance arrays — orders of magnitude less RAM.

Additionally:
  - Load only needed columns (saves ~40% DataFrame RAM)
  - Process S1 kneighbors in batches of 50K rows
  - Reduced max_features from 300K → 100K
  - Disabled bigrams (ngram_range=(1,1)) to further shrink matrix width
"""

import gc
import sys
import time
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout
)
log = logging.getLogger(__name__)

# ─── Columns we actually need (drop addr_norm, etc. to save RAM) ───────────
COLS = ['entity_id', 'country', 'name_norm', 'composite_key']


def get_top_k_neighbors(
    query_matrix,       # sparse (n_s1, vocab)
    index_matrix,       # sparse (n_cand, vocab)
    K: int,
    batch_size: int = 50_000
) -> np.ndarray:
    """
    Memory-safe Top-K nearest neighbor retrieval using sklearn NearestNeighbors.

    NearestNeighbors.kneighbors() NEVER builds the full (n_s1 × n_cand) matrix.
    It computes cosine similarity row-by-row, keeping only the top-K per row.

    We additionally batch the query side (50K at a time) so the result array
    is also bounded: 50_000 × K × 8 bytes = ~20 MB per batch (very manageable).
    """
    # Clamp K to available candidates
    K = min(K, index_matrix.shape[0])

    nn = NearestNeighbors(
        n_neighbors=K,
        metric='cosine',
        algorithm='brute',
        n_jobs=-1          # use all CPU cores
    )
    nn.fit(index_matrix)

    all_indices = []
    n_queries = query_matrix.shape[0]

    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        batch = query_matrix[start:end]
        _, indices = nn.kneighbors(batch)   # shape (batch_size, K) — small!
        all_indices.append(indices)
        log.info(f"    Batch {start//batch_size + 1}: {start} → {end} queries done")

    return np.vstack(all_indices)   # shape (n_queries, K)


def build_tfidf(fit_corpus, transform_corpus, max_features=100_000):
    """
    Fit TF-IDF on candidates, transform both candidates and S1.
    Uses unigrams only + max_df=0.05 to keep matrix very sparse.
    """
    vec = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 1),     # unigrams only — smallest possible matrix width
        min_df=3,               # ignore very rare tokens (noise)
        max_df=0.05,            # ignore hyper-common tokens (inc, llc, road, street)
        max_features=max_features,
        dtype=np.float32,       # half the memory of float64
        sublinear_tf=True,      # log(1+tf) — compresses high-freq tokens
    )
    fit_mat   = vec.fit_transform(fit_corpus)
    trans_mat = vec.transform(transform_corpus)
    del vec
    gc.collect()
    return fit_mat, trans_mat


def run_blocking(
    split: str,
    data_dir: Path,
    output_file: Path,
    top_k: int = 50,
    max_features: int = 100_000,
    batch_size: int = 50_000,
):
    t_start = time.time()
    log.info(f"Starting Memory-Safe Blocking | split={split} | top_k={top_k}")

    s1_path = data_dir / split / f"preprocessed_{split}_source1.tsv"
    s2_path = data_dir / split / f"preprocessed_{split}_source2.tsv"
    s3_path = data_dir / split / f"preprocessed_{split}_source3.tsv"

    log.info("Loading S1 (only needed columns)...")
    s1 = pd.read_csv(s1_path, sep='\t', usecols=COLS, dtype=str).fillna("")
    countries = sorted(c for c in s1['country'].unique() if c)
    log.info(f"S1 loaded: {len(s1):,} rows | Countries: {countries}")

    k_name = max(1, top_k // 2)
    k_comp = top_k - k_name      # remaining budget for composite key

    final_results = []

    for country in countries:
        log.info(f"\n{'='*55}")
        log.info(f"Country: {country}")
        log.info(f"{'='*55}")

        s1_c  = s1[s1['country'] == country]
        s1_ids = s1_c['entity_id'].values
        log.info(f"S1 for {country}: {len(s1_c):,} entities")

        # ── Load candidates fresh per country to avoid holding all 10M rows ──
        log.info("Loading S2 (filtered to country)...")
        s2 = pd.read_csv(s2_path, sep='\t', usecols=COLS, dtype=str).fillna("")
        s2 = s2[s2['country'] == country]
        log.info("Loading S3 (filtered to country)...")
        s3 = pd.read_csv(s3_path, sep='\t', usecols=COLS, dtype=str).fillna("")
        s3 = s3[s3['country'] == country]

        cand_c   = pd.concat([s2, s3], ignore_index=True)
        cand_ids = cand_c['entity_id'].values
        log.info(f"Candidates for {country}: {len(cand_c):,} entities")
        del s2, s3
        gc.collect()

        if s1_c.empty or cand_c.empty:
            # Emit empty rows for completeness
            for s1_id in s1_ids:
                final_results.append({'source1_entity_id': s1_id, 'candidate_entity_ids': ''})
            continue

        # ─────────────────────────────────────────────
        # BLOCKING KEY 1: Business Name TF-IDF
        # ─────────────────────────────────────────────
        log.info(f"\n[Key 1] Name TF-IDF | Top-{k_name} per entity")
        cand_name_mat, s1_name_mat = build_tfidf(
            cand_c['name_norm'], s1_c['name_norm'], max_features
        )
        log.info(f"  Matrix: cand={cand_name_mat.shape}, s1={s1_name_mat.shape}")

        top_name_idx = get_top_k_neighbors(
            s1_name_mat, cand_name_mat, K=k_name, batch_size=batch_size
        )
        del cand_name_mat, s1_name_mat
        gc.collect()

        # ─────────────────────────────────────────────
        # BLOCKING KEY 2: Composite Key TF-IDF (Name + Address)
        # ─────────────────────────────────────────────
        log.info(f"\n[Key 2] Composite TF-IDF | Top-{k_comp} per entity")
        cand_comp_mat, s1_comp_mat = build_tfidf(
            cand_c['composite_key'], s1_c['composite_key'], max_features
        )
        log.info(f"  Matrix: cand={cand_comp_mat.shape}, s1={s1_comp_mat.shape}")

        top_comp_idx = get_top_k_neighbors(
            s1_comp_mat, cand_comp_mat, K=k_comp, batch_size=batch_size
        )
        del cand_comp_mat, s1_comp_mat
        gc.collect()

        # ─────────────────────────────────────────────
        # UNION: Merge both candidate sets per S1 entity
        # ─────────────────────────────────────────────
        log.info("\nMerging candidate sets...")
        for i, s1_id in enumerate(s1_ids):
            idx_union = set(top_name_idx[i].tolist()) | set(top_comp_idx[i].tolist())
            matched = cand_ids[sorted(idx_union)]
            final_results.append({
                'source1_entity_id': s1_id,
                'candidate_entity_ids': ','.join(matched)
            })

        del top_name_idx, top_comp_idx, cand_c
        gc.collect()

        elapsed = (time.time() - t_start) / 60
        log.info(f"Country {country} done | Total elapsed: {elapsed:.1f} min")

    # ── Save output ────────────────────────────────────────────────────────
    log.info("\nSaving candidate pairs...")
    output_file.parent.mkdir(exist_ok=True, parents=True)
    out_df = pd.DataFrame(final_results)

    # Ensure every S1 row is present (even those with no candidates)
    s1_ids_all = s1['entity_id'].values
    out_df = out_df.set_index('source1_entity_id').reindex(s1_ids_all).fillna('').reset_index()
    out_df.columns = ['source1_entity_id', 'candidate_entity_ids']

    out_df.to_csv(output_file, sep='\t', index=False)

    total_min = (time.time() - t_start) / 60
    avg_cands = out_df['candidate_entity_ids'].apply(
        lambda x: len(x.split(',')) if x else 0
    ).mean()
    log.info(f"\nDone! {len(out_df):,} rows saved to {output_file}")
    log.info(f"Avg candidates per S1 entity: {avg_cands:.1f}")
    log.info(f"Total time: {total_min:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Memory-Safe Multi-Key Blocking")
    parser.add_argument('--data-dir',     default='dataset/preprocessed')
    parser.add_argument('--out-dir',      default='outputs')
    parser.add_argument('--split',        required=True, choices=['train', 'test'])
    parser.add_argument('--top-k',        type=int, default=50,
                        help="Total candidates per S1 entity (split across both keys)")
    parser.add_argument('--max-features', type=int, default=100_000,
                        help="TF-IDF vocabulary size (default 100K)")
    parser.add_argument('--batch-size',   type=int, default=50_000,
                        help="S1 batch size for kneighbors (default 50K)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    output_file = out_dir / ("candidate_pairs.tsv" if args.split == 'test'
                             else f"candidate_pairs_{args.split}.tsv")

    run_blocking(
        split=args.split,
        data_dir=data_dir,
        output_file=output_file,
        top_k=args.top_k,
        max_features=args.max_features,
        batch_size=args.batch_size,
    )
