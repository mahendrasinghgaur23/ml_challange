"""
Stage 2: Multi-Key Blocking
=============================================================
This stage reduces the search space from O(N^2) to a small set of candidates 
per Source 1 entity. 

We use TF-IDF + Cosine Similarity implemented via highly optimized 
sparse matrix multiplication to find the Top-K candidates very quickly.
We also use Exact Country filtering to ensure we don't compare across countries.

Optimized for Kaggle (30GB RAM limit):
- Uses chunked sparse matrix multiplication (A_chunk.dot(B.T)).
- Employs max_df to prevent massive dense intermediate matrices.
- Aggressively frees memory between Name and Composite key processing.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def get_top_k_sparse(A: csr_matrix, B: csr_matrix, K: int = 30) -> tuple:
    """
    Computes top K cosine similarities between rows of A (queries) and B (database).
    A and B should be L2-normalized sparse matrices.
    Returns:
        indices: (A.shape[0], K) matrix of B's row indices
    """
    # Chunk size is critical for RAM limit. Smaller chunks = less memory for intermediate dot products.
    # Reduced to 100 to guarantee peak RAM stays well below 22GB.
    # A 100 x 6,000,000 dense float32 array is only ~2.4 GB.
    chunk_size = 100 
    all_indices = []
    
    total_rows = A.shape[0]
    logged_pct = -1

    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        pct = int(100 * end_idx / total_rows)
        
        if pct // 10 > logged_pct // 10:
            log.info(f"    Querying chunk: {end_idx:>7,}/{total_rows:,} ({pct}%)")
            logged_pct = pct

        A_chunk = A[start_idx:end_idx]
        
        # Cosine similarity (A and B are L2 normalized, so dot product == cosine sim)
        sim_chunk = A_chunk.dot(B.T) 
        
        if sim_chunk.shape[1] > K:
            # Convert to dense (Now only 2.4GB max)
            sim_dense = sim_chunk.toarray()
            # Find indices of top K elements
            top_k_idx = np.argpartition(sim_dense, -K, axis=1)[:, -K:]
            
            # Sort the top K to get them in descending order of similarity
            for i in range(top_k_idx.shape[0]):
                top_k_idx[i] = top_k_idx[i][np.argsort(-sim_dense[i, top_k_idx[i]])]
            del sim_dense
        else:
            sim_dense = sim_chunk.toarray()
            top_k_idx = np.argsort(-sim_dense, axis=1)
            del sim_dense
            
        all_indices.append(top_k_idx)
    
    return np.vstack(all_indices)


def run_blocking(train_or_test: str, data_dir: Path, output_file: Path, top_k: int = 30):
    """Run TF-IDF blocking on preprocessed data."""
    t_start = time.time()
    log.info(f"Starting blocking for {train_or_test} split... (Strictly <22GB RAM)")
    
    # Load data (Updated for Kaggle Paths)
    # The user path looks like: /kaggle/input/datasets/mahendrasinghgaur/preprocessed-train-data/preprocessed_train_source1.tsv
    # Wait, the path might not have a 'train' subdirectory anymore based on user prompt. Let's handle it carefully.
    
    s1_path = data_dir / f"preprocessed_{train_or_test}_source1.tsv"
    s2_path = data_dir / f"preprocessed_{train_or_test}_source2.tsv"
    s3_path = data_dir / f"preprocessed_{train_or_test}_source3.tsv"
    
    log.info("Loading preprocessed TSVs...")
    s1 = pd.read_csv(s1_path, sep='\t', dtype=str).fillna("")
    s2 = pd.read_csv(s2_path, sep='\t', dtype=str).fillna("")
    s3 = pd.read_csv(s3_path, sep='\t', dtype=str).fillna("")
    
    log.info(f"Loaded S1: {len(s1):,}, S2: {len(s2):,}, S3: {len(s3):,}")
    
    # We will block S1 against (S2 + S3) combined to get global top K
    candidates_df = pd.concat([s2, s3], ignore_index=True)
    
    final_results = []
    
    # Process country by country to avoid cross-country matches
    countries = s1['country'].unique()
    
    for country in countries:
        if not country: continue
        log.info(f"\n{'='*60}")
        log.info(f"Processing country: {country}")
        log.info(f"{'='*60}")
        
        s1_c = s1[s1['country'] == country].copy()
        cand_c = candidates_df[candidates_df['country'] == country].copy()
        
        if s1_c.empty or cand_c.empty:
            for eid in s1_c["entity_id"]:
                final_results.append({"source1_entity_id": eid, "candidate_entity_ids": ""})
            continue
            
        cand_ids = cand_c['entity_id'].values
        s1_ids = s1_c['entity_id'].values
        k_per_strategy = max(1, top_k // 2)
        
        log.info(f"  S1={len(s1_ids):,}  |  Candidates={len(cand_ids):,}")

        # ---------------------------------------------------------
        # 1. Name Blocking
        # ---------------------------------------------------------
        log.info("Building TF-IDF vectorizer for Names...")
        # Switched to word analyzer and np.float32 to drastically reduce memory usage.
        # max_df=0.02 removes top 2% words. max_features=100_000 keeps vocab small.
        name_vec = TfidfVectorizer(
            analyzer='word', 
            ngram_range=(1, 2), 
            min_df=5, 
            max_df=0.02, 
            max_features=100_000, 
            dtype=np.float32
        )
        
        cand_name_tfidf = name_vec.fit_transform(cand_c['name_norm'])
        s1_name_tfidf = name_vec.transform(s1_c['name_norm'])
        
        log.info(f" -> Computing Name Top-{k_per_strategy} nearest neighbors via Sparse Dot Product...")
        top_name_idx = get_top_k_sparse(s1_name_tfidf, cand_name_tfidf, K=k_per_strategy)
        
        # Aggressive Memory Cleanup before Composite key processing
        del name_vec
        del cand_name_tfidf
        del s1_name_tfidf
        import gc
        gc.collect()
        
        # ---------------------------------------------------------
        # 2. Composite Key Blocking
        # ---------------------------------------------------------
        log.info("Building TF-IDF vectorizer for Composite Key (Name + Address)...")
        comp_vec = TfidfVectorizer(
            analyzer='word', 
            ngram_range=(1, 2), 
            min_df=5, 
            max_df=0.02, 
            max_features=100_000, 
            dtype=np.float32
        )
        
        cand_comp_tfidf = comp_vec.fit_transform(cand_c['composite_key'])
        s1_comp_tfidf = comp_vec.transform(s1_c['composite_key'])
        
        log.info(f" -> Computing Composite Top-{k_per_strategy} nearest neighbors via Sparse Dot Product...")
        top_comp_idx = get_top_k_sparse(s1_comp_tfidf, cand_comp_tfidf, K=k_per_strategy)
        
        # Aggressive Memory Cleanup
        del comp_vec
        del cand_comp_tfidf
        del s1_comp_tfidf
        gc.collect()
        
        # ---------------------------------------------------------
        # Combine Candidates
        # ---------------------------------------------------------
        log.info("Combining candidate lists...")
        for i, s1_id in enumerate(s1_ids):
            # Union of indices from both strategies
            match_idx = set(top_name_idx[i]).union(set(top_comp_idx[i]))
            
            # Map indices back to actual entity_ids
            matched_entity_ids = cand_ids[list(match_idx)]
            
            final_results.append({
                'source1_entity_id': s1_id,
                'candidate_entity_ids': ",".join(matched_entity_ids)
            })
            
        del top_name_idx
        del top_comp_idx
        gc.collect()
            
    # Save results
    log.info("Saving candidate pairs...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_df = pd.DataFrame(final_results)
    output_df.to_csv(output_file, sep='\t', index=False)
    
    log.info(f"Done! Saved {len(output_df):,} rows to {output_file}. Time: {(time.time()-t_start)/60:.1f} mins.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/kaggle/input/datasets/mahendrasinghgaur/preprocessed-train-data', help="Path to preprocessed data")
    parser.add_argument('--out-dir', type=str, default='/kaggle/working/project/outputs', help="Output directory")
    parser.add_argument('--split', type=str, required=True, choices=['train', 'test'], help="train or test split")
    parser.add_argument('--top-k', type=int, default=50, help="Total candidates to retrieve per S1 entity")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    
    output_file = out_dir / f"candidate_pairs_{args.split}.tsv"
    if args.split == 'test':
        output_file = out_dir / "candidate_pairs.tsv"
        
    run_blocking(args.split, data_dir, output_file, top_k=args.top_k)
