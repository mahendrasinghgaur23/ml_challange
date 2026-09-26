"""
Stage 2: Multi-Key Blocking
=============================================================
This stage reduces the search space from O(N^2) to a small set of candidates 
per Source 1 entity. 

We use TF-IDF + Cosine Similarity (as suggested!) implemented via highly optimized 
sparse matrix multiplication to find the Top-K candidates very quickly.
We also use Exact Country filtering to ensure we don't compare across countries.
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
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def get_top_k_sparse(A: csr_matrix, B: csr_matrix, K: int = 30) -> tuple:
    """
    Computes top K cosine similarities between rows of A (queries) and B (database).
    A and B should be L2-normalized sparse matrices.
    Returns:
        indices: (A.shape[0], K) matrix of B's row indices
        scores: (A.shape[0], K) matrix of cosine similarity scores
    """
    # To handle memory, we process A in chunks
    # Reduced chunk_size to 500 to prevent massive dense output matrices during dot product
    chunk_size = 500
    all_indices = []
    # all_scores = [] # We don't strictly need scores for the final output, just indices
    
    for start_idx in range(0, A.shape[0], chunk_size):
        end_idx = min(start_idx + chunk_size, A.shape[0])
        A_chunk = A[start_idx:end_idx]
        
        # Cosine similarity (since A and B are L2 normalized, dot product == cosine sim)
        sim_chunk = A_chunk.dot(B.T) 
        
        # Get top K indices for each row in the chunk
        # argpartition is much faster than full sort
        if sim_chunk.shape[1] > K:
            # np.argpartition on sparse matrix requires converting to dense or using specialized sparse routines.
            # Convert chunk to dense (safe because chunk is small and K is small)
            sim_dense = sim_chunk.toarray()
            top_k_idx = np.argpartition(sim_dense, -K, axis=1)[:, -K:]
            
            # Sort the top K by actual score (argpartition doesn't guarantee order)
            for i in range(top_k_idx.shape[0]):
                top_k_idx[i] = top_k_idx[i][np.argsort(-sim_dense[i, top_k_idx[i]])]
        else:
            sim_dense = sim_chunk.toarray()
            top_k_idx = np.argsort(-sim_dense, axis=1)
            
        all_indices.append(top_k_idx)
    
    return np.vstack(all_indices)


def run_blocking(train_or_test: str, data_dir: Path, output_file: Path, top_k: int = 30):
    """Run TF-IDF blocking on preprocessed data."""
    t_start = time.time()
    log.info(f"Starting blocking for {train_or_test} split...")
    
    # Load data
    s1_path = data_dir / train_or_test / f"preprocessed_{train_or_test}_source1.tsv"
    s2_path = data_dir / train_or_test / f"preprocessed_{train_or_test}_source2.tsv"
    s3_path = data_dir / train_or_test / f"preprocessed_{train_or_test}_source3.tsv"
    
    log.info("Loading preprocessed TSVs (this might take a minute)...")
    s1 = pd.read_csv(s1_path, sep='\t', dtype=str).fillna("")
    s2 = pd.read_csv(s2_path, sep='\t', dtype=str).fillna("")
    s3 = pd.read_csv(s3_path, sep='\t', dtype=str).fillna("")
    
    log.info(f"Loaded S1: {len(s1)}, S2: {len(s2)}, S3: {len(s3)}")
    
    # We will block S1 against (S2 + S3) combined to get global top K
    candidates_df = pd.concat([s2, s3], ignore_index=True)
    
    final_results = []
    
    # Process country by country to avoid cross-country matches (which are impossible)
    countries = s1['country'].unique()
    
    for country in countries:
        if not country: continue
        log.info(f"--- Processing country: {country} ---")
        
        s1_c = s1[s1['country'] == country].copy()
        cand_c = candidates_df[candidates_df['country'] == country].copy()
        
        if s1_c.empty or cand_c.empty:
            continue
            
        cand_ids = cand_c['entity_id'].values
        s1_ids = s1_c['entity_id'].values
        k_per_strategy = max(1, top_k // 2)
        
        # 1. Name Blocking
        log.info("Building TF-IDF vectorizer for Names...")
        # Switched to word analyzer and np.float32 to drastically reduce memory usage
        # Added max_df=0.05 to eliminate super-common words (like inc, llc, road) which make the dot product dense!
        name_vec = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=5, max_df=0.05, max_features=300000, dtype=np.float32)
        
        cand_name_tfidf = name_vec.fit_transform(cand_c['name_norm'])
        s1_name_tfidf = name_vec.transform(s1_c['name_norm'])
        
        log.info(f" -> Computing Name Top-{k_per_strategy} nearest neighbors...")
        top_name_idx = get_top_k_sparse(s1_name_tfidf, cand_name_tfidf, K=k_per_strategy)
        
        # Aggressive Memory Cleanup
        del name_vec
        del cand_name_tfidf
        del s1_name_tfidf
        import gc
        gc.collect()
        
        # 2. Composite Key Blocking
        log.info("Building TF-IDF vectorizer for Composite Key (Name + Address)...")
        comp_vec = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=5, max_df=0.05, max_features=300000, dtype=np.float32)
        
        cand_comp_tfidf = comp_vec.fit_transform(cand_c['composite_key'])
        s1_comp_tfidf = comp_vec.transform(s1_c['composite_key'])
        
        log.info(f" -> Computing Composite Top-{k_per_strategy} nearest neighbors...")
        top_comp_idx = get_top_k_sparse(s1_comp_tfidf, cand_comp_tfidf, K=k_per_strategy)
        
        # Aggressive Memory Cleanup
        del comp_vec
        del cand_comp_tfidf
        del s1_comp_tfidf
        gc.collect()
        
        # Combine candidates
        log.info("Combining candidate lists...")
        for i, s1_id in enumerate(s1_ids):
            # Get integer indices of matches from cand_c
            match_idx = set(top_name_idx[i]).union(set(top_comp_idx[i]))
            
            # Map integer indices back to actual entity_ids
            matched_entity_ids = cand_ids[list(match_idx)]
            
            # Join with comma
            final_results.append({
                'source1_entity_id': s1_id,
                'candidate_entity_ids': ",".join(matched_entity_ids)
            })
            
        del top_name_idx
        del top_comp_idx
        gc.collect()
            
    # Save results
    log.info("Saving candidate pairs...")
    output_df = pd.DataFrame(final_results)
    output_df.to_csv(output_file, sep='\t', index=False)
    
    log.info(f"Done! Saved {len(output_df)} rows to {output_file}. Time: {(time.time()-t_start)/60:.1f} mins.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='dataset/preprocessed', help="Path to preprocessed data")
    parser.add_argument('--out-dir', type=str, default='outputs', help="Output directory")
    parser.add_argument('--split', type=str, required=True, choices=['train', 'test'], help="train or test split")
    parser.add_argument('--top-k', type=int, default=50, help="Total candidates to retrieve per S1 entity")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    output_file = out_dir / f"candidate_pairs_{args.split}.tsv"
    if args.split == 'test':
        # the final submission uses specific filename
        output_file = out_dir / "candidate_pairs.tsv"
        
    run_blocking(args.split, data_dir, output_file, top_k=args.top_k)
