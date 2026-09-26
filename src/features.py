"""
Stage 3: Feature Engineering
=============================================================
Computes pairwise similarity features between Source 1 entities and 
their retrieved candidates from Stage 2.

Features include:
- Jaccard Similarity (Token level)
- Levenshtein Distance / Ratio (via rapidfuzz)
- Token Sort/Set Ratios (handles word reordering)
- Length differences

Outputs a dataset ready for LightGBM training/inference.
To handle scale (potentially 100M+ pairs), we process in chunks and save to Parquet.
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from rapidfuzz import fuzz, distance

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def compute_jaccard(str1: str, str2: str) -> float:
    """Token-level Jaccard similarity."""
    if not str1 and not str2:
        return 1.0
    if not str1 or not str2:
        return 0.0
    
    set1 = set(str1.split())
    set2 = set(str2.split())
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns: 
    [s1_name, s1_addr, cand_name, cand_addr]
    Compute and return a DataFrame of features.
    """
    # Initialize dictionary for new features
    feats = {}
    
    # Fill NAs with empty strings to avoid errors
    n1 = df['s1_name'].fillna("").astype(str).values
    a1 = df['s1_addr'].fillna("").astype(str).values
    n2 = df['cand_name'].fillna("").astype(str).values
    a2 = df['cand_addr'].fillna("").astype(str).values

    log.info("Computing Name features...")
    # Levenshtein Ratio (0-100) -> scale to 0-1
    feats['name_levenshtein'] = [fuzz.ratio(x, y) / 100.0 for x, y in zip(n1, n2)]
    # Token Sort Ratio (handles word reordering, e.g., "McDonalds Corp" vs "Corp McDonalds")
    feats['name_token_sort'] = [fuzz.token_sort_ratio(x, y) / 100.0 for x, y in zip(n1, n2)]
    # Token Set Ratio (handles subsets, e.g., "McDonalds" vs "McDonalds Corporation")
    feats['name_token_set'] = [fuzz.token_set_ratio(x, y) / 100.0 for x, y in zip(n1, n2)]
    # Jaccard Token Similarity
    feats['name_jaccard'] = [compute_jaccard(x, y) for x, y in zip(n1, n2)]
    # Length features
    feats['name_len_diff'] = np.abs(np.array([len(x) for x in n1]) - np.array([len(y) for y in n2]))
    
    log.info("Computing Address features...")
    feats['addr_levenshtein'] = [fuzz.ratio(x, y) / 100.0 for x, y in zip(a1, a2)]
    feats['addr_token_sort'] = [fuzz.token_sort_ratio(x, y) / 100.0 for x, y in zip(a1, a2)]
    feats['addr_token_set'] = [fuzz.token_set_ratio(x, y) / 100.0 for x, y in zip(a1, a2)]
    feats['addr_jaccard'] = [compute_jaccard(x, y) for x, y in zip(a1, a2)]
    
    # Combined strings for global context
    c1 = [f"{x} {y}".strip() for x, y in zip(n1, a1)]
    c2 = [f"{x} {y}".strip() for x, y in zip(n2, a2)]
    log.info("Computing Composite features...")
    feats['comp_levenshtein'] = [fuzz.ratio(x, y) / 100.0 for x, y in zip(c1, c2)]
    feats['comp_jaccard'] = [compute_jaccard(x, y) for x, y in zip(c1, c2)]
    
    return pd.DataFrame(feats, index=df.index)


def run_feature_engineering(split: str, data_dir: Path, output_dir: Path, cand_file: Path):
    t_start = time.time()
    log.info(f"Starting Feature Engineering for {split} split...")
    
    # 1. Load Preprocessed Data
    log.info("Loading preprocessed datasets...")
    s1 = pd.read_csv(data_dir / split / f"preprocessed_{split}_source1.tsv", sep='\t', dtype=str)
    s2 = pd.read_csv(data_dir / split / f"preprocessed_{split}_source2.tsv", sep='\t', dtype=str)
    s3 = pd.read_csv(data_dir / split / f"preprocessed_{split}_source3.tsv", sep='\t', dtype=str)
    
    # Prepare lookups
    s1_lookup = s1.set_index('entity_id')[['name_norm', 'addr_norm']]
    cand_lookup = pd.concat([s2, s3]).set_index('entity_id')[['name_norm', 'addr_norm']]
    
    # 2. Load Ground Truth (if training)
    gt_dict = {}
    if split == 'train':
        log.info("Loading Ground Truth...")
        gt = pd.read_csv(data_dir.parent / "train" / "train_ground_truth.tsv", sep='\t', dtype=str)
        # Create a dictionary mapping s1_id -> set of true matching cand_ids
        for _, row in gt.iterrows():
            s1_id = str(row['source1_entity_id'])
            matches = str(row['matched_entity_ids']).split(',') if pd.notna(row['matched_entity_ids']) and row['matched_entity_ids'] else []
            gt_dict[s1_id] = set(m.strip() for m in matches if m.strip())
            
    # 3. Load Candidates and Flatten
    log.info(f"Loading candidate pairs from {cand_file}...")
    candidates = pd.read_csv(cand_file, sep='\t', dtype=str)
    
    log.info("Flattening candidates into pairwise format (this may take a moment)...")
    pairs = []
    for _, row in candidates.iterrows():
        s1_id = str(row['source1_entity_id'])
        cands = str(row['candidate_entity_ids']).split(',') if pd.notna(row['candidate_entity_ids']) and row['candidate_entity_ids'] else []
        for c in cands:
            c = c.strip()
            if c:
                # Assign label immediately if it's the train split
                label = 1 if (split == 'train' and c in gt_dict.get(s1_id, set())) else 0
                pairs.append({'s1_id': s1_id, 'cand_id': c, 'label': label})
                
    df_pairs = pd.DataFrame(pairs)
    log.info(f"Total pairs to process: {len(df_pairs):,}")
    
    if len(df_pairs) == 0:
        log.warning("No pairs found! Exiting.")
        return

    # 4. Process in Chunks
    chunk_size = 2_000_000  # 2 million pairs per chunk to manage memory safely
    num_chunks = int(np.ceil(len(df_pairs) / chunk_size))
    
    out_path_dir = output_dir / f"features_{split}"
    out_path_dir.mkdir(exist_ok=True, parents=True)
    
    for i in range(num_chunks):
        log.info(f"--- Processing Chunk {i+1} of {num_chunks} ---")
        chunk = df_pairs.iloc[i*chunk_size : (i+1)*chunk_size].copy()
        
        # Merge attributes
        log.info("Merging attributes...")
        chunk = chunk.join(s1_lookup.rename(columns={'name_norm': 's1_name', 'addr_norm': 's1_addr'}), on='s1_id', how='left')
        chunk = chunk.join(cand_lookup.rename(columns={'name_norm': 'cand_name', 'addr_norm': 'cand_addr'}), on='cand_id', how='left')
        
        # Compute features
        feats_df = extract_features(chunk)
        
        # Combine IDs, features, and label
        final_chunk = pd.concat([chunk[['s1_id', 'cand_id', 'label']], feats_df], axis=1)
        
        # Save chunk
        chunk_file = out_path_dir / f"chunk_{i:03d}.parquet"
        final_chunk.to_parquet(chunk_file, index=False)
        log.info(f"Saved chunk to {chunk_file}")
        
    log.info(f"Feature Engineering complete! Total time: {(time.time() - t_start)/60:.1f} mins.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='dataset/preprocessed', help="Path to preprocessed data")
    parser.add_argument('--out-dir', type=str, default='outputs', help="Output directory for features")
    parser.add_argument('--cand-file', type=str, required=True, help="Path to candidate_pairs.tsv from Stage 2")
    parser.add_argument('--split', type=str, required=True, choices=['train', 'test'], help="train or test split")
    args = parser.parse_args()
    
    run_feature_engineering(args.split, Path(args.data_dir), Path(args.out_dir), Path(args.cand_file))
