"""
Stage 1: Preprocessing — Business Entity Resolution Pipeline
=============================================================
Reads each source TSV in configurable chunks, applies text normalization,
and writes clean preprocessed TSVs to `dataset/preprocessed/`.

Normalization steps (applied to both business_name and business_address):
  1. Unicode safety  — decode safely, keep Devanagari/Tamil as-is for now
  2. Missing-value handling — treat NaN / literal 'nan' as empty string
  3. Lowercase
  4. Legal suffix canonicalization — e.g. "Pvt" → "private", "Ltd" → "limited"
  5. Ampersand expansion — "&" → "and"
  6. Address abbreviation expansion — "Rd" → "road", "St" → "street", etc.
  7. Punctuation removal — keep alphanumeric, spaces, hyphens, slashes
  8. Extra whitespace collapse
  9. Composite key — name_norm + " " + addr_norm  (used later in blocking)

Output columns (added):
  - name_norm       : normalized business name
  - addr_norm       : normalized business address
  - composite_key   : name_norm + " " + addr_norm
"""

import os
import re
import sys
import time
import logging
import argparse
from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Lookup Tables
# ─────────────────────────────────────────────────────────────────────────────

# Legal/corporate suffix canonicalization  (pattern → replacement)
# Order matters: longer matches first to avoid partial replacements.
LEGAL_SUFFIX_MAP = [
    # Pvt variants  → "private"
    (r'\bpvt\.?\b',         'private'),
    # Ltd variants  → "limited"
    (r'\bltd\.?\b',         'limited'),
    # Corp variants → "corporation"
    (r'\bcorp\.?\b',        'corporation'),
    # Inc variants  → "incorporated"
    (r'\binc\.?\b',         'incorporated'),
    # Co. → company  (must come BEFORE general 'co' to avoid over-matching)
    (r'\bco\.?\b',          'company'),
    # LLP / LTD LLP variants (India)
    (r'\bllp\.?\b',         'llp'),
    # LLC
    (r'\bllc\.?\b',         'llc'),
    # Enterprises shortcuts
    (r'\bentps\.?\b',       'enterprises'),
    (r'\bentp\.?\b',        'enterprises'),
    # Assoc → associates
    (r'\bassoc\.?\b',       'associates'),
    # Mfg → manufacturing
    (r'\bmfg\.?\b',         'manufacturing'),
    # Intl → international
    (r'\bintl\.?\b',        'international'),
    # Bros → brothers
    (r'\bbros\.?\b',        'brothers'),
    # Mgmt → management
    (r'\bmgmt\.?\b',        'management'),
    # Dept → department
    (r'\bdept\.?\b',        'department'),
]

# Pre-compile for speed
LEGAL_SUFFIX_PATTERNS = [
    (re.compile(pat, flags=re.IGNORECASE | re.UNICODE), repl)
    for pat, repl in LEGAL_SUFFIX_MAP
]

# Address abbreviation expansion  (word-boundary pattern → replacement)
ADDRESS_ABBR_MAP = [
    # Street types
    (r'\bst\.?\b',    'street'),
    (r'\bave\.?\b',   'avenue'),
    (r'\bavenue\b',   'avenue'),        # already full — keep
    (r'\bblvd\.?\b',  'boulevard'),
    (r'\brd\.?\b',    'road'),
    (r'\bdr\.?\b',    'drive'),
    (r'\bln\.?\b',    'lane'),
    (r'\bct\.?\b',    'court'),
    (r'\bpl\.?\b',    'place'),
    (r'\bpkwy\.?\b',  'parkway'),
    (r'\bhwy\.?\b',   'highway'),
    (r'\bfwy\.?\b',   'freeway'),
    (r'\bexpy\.?\b',  'expressway'),
    (r'\bsq\.?\b',    'square'),
    (r'\bter\.?\b',   'terrace'),
    (r'\bterr\.?\b',  'terrace'),
    (r'\bcir\.?\b',   'circle'),
    (r'\bxing\.?\b',  'crossing'),
    (r'\bpike\.?\b',  'pike'),
    # Directional
    (r'\bn\.?\b',     'north'),
    (r'\bs\.?\b',     'south'),
    (r'\be\.?\b',     'east'),
    (r'\bw\.?\b',     'west'),
    (r'\bne\.?\b',    'northeast'),
    (r'\bnw\.?\b',    'northwest'),
    (r'\bse\.?\b',    'southeast'),
    (r'\bsw\.?\b',    'southwest'),
    # Unit/suite/building
    (r'\bapt\.?\b',   'apartment'),
    (r'\bste\.?\b',   'suite'),
    (r'\bbldg\.?\b',  'building'),
    (r'\bfl\.?\b',    'floor'),
    (r'\bflr\.?\b',   'floor'),
    (r'\bunit\.?\b',  'unit'),
    # Common US address tokens
    (r'\bmt\.?\b',    'mount'),
    (r'\bft\.?\b',    'fort'),
    (r'\bpt\.?\b',    'point'),
    # Indian address tokens
    (r'\bno\.?\b',    'number'),
    (r'\bhn\.?\b',    'house number'),
    (r'\bh\.no\.?\b', 'house number'),
    (r'\bph\.?\b',    'phase'),
    (r'\bsec\.?\b',   'sector'),
    (r'\bsect\.?\b',  'sector'),
    (r'\bsoc\.?\b',   'society'),
    (r'\bnr\.?\b',    'near'),
    (r'\bopp\.?\b',   'opposite'),
]

ADDRESS_ABBR_PATTERNS = [
    (re.compile(pat, flags=re.IGNORECASE | re.UNICODE), repl)
    for pat, repl in ADDRESS_ABBR_MAP
]

# Regex for collapsing whitespace and stripping non-useful punctuation.
# IMPORTANT: We use \x00-\x7F to limit removal to ASCII punctuation only,
# so Devanagari, Tamil, Arabic, French accented chars etc. are preserved.
# Keep: alphanumeric (all scripts), spaces, hyphens (-), slashes (/)
RE_PUNCTUATION = re.compile(r"[^\w\s\-/]", flags=re.UNICODE)

# Collapse dotted single-char abbreviations like A.B.C. → ABC (before lowercasing)
# Matches sequences like "A.B." or "U.S.A" etc.
RE_DOTTED_ABBR = re.compile(r"\b([A-Za-z])\.(?=[A-Za-z]\.?)", flags=re.UNICODE)

RE_WHITESPACE  = re.compile(r"\s+")

# Detect literal "nan" strings
RE_NAN_LITERAL = re.compile(r"^\s*nan\s*$", flags=re.IGNORECASE)

# Strip leading/trailing hyphens and slashes after normalization
RE_EDGE_PUNCT  = re.compile(r"^[\-/\s]+|[\-/\s]+$")


# ─────────────────────────────────────────────────────────────────────────────
# Core Normalization Functions
# ─────────────────────────────────────────────────────────────────────────────

def _safe_str(val) -> str:
    """Convert a cell value to string, treating NaN / None / literal 'nan' as ''."""
    if val is None or (isinstance(val, float) and val != val):  # NaN check
        return ''
    s = str(val).strip()
    if RE_NAN_LITERAL.match(s):
        return ''
    return s


def normalize_business_name(name_raw: str) -> str:
    """
    Normalize a business name string.

    Steps:
      1. Safe string conversion
      2. Collapse dotted single-char abbreviations: A.B.C. → ABC
      3. Lowercase
      4. Ampersand → "and"
      5. Legal suffix canonicalization
      6. Remove ASCII punctuation (keep hyphens/slashes; preserve non-ASCII scripts)
      7. Strip edge hyphens/slashes
      8. Collapse whitespace
    """
    text = _safe_str(name_raw)
    if not text:
        return ''

    # Step 2: Collapse dotted abbreviations BEFORE lowercasing (e.g. A.B.C. → ABC)
    text = RE_DOTTED_ABBR.sub(r'\1', text)

    # Step 3: Lowercase
    text = text.lower()

    # Step 4: Expand & → and  (before punct removal)
    text = text.replace('&', ' and ')

    # Step 5: Canonicalize legal suffixes
    for pattern, replacement in LEGAL_SUFFIX_PATTERNS:
        text = pattern.sub(replacement, text)

    # Step 6: Remove ASCII punctuation only (preserves Devanagari, Tamil, French, etc.)
    # \w in Python Unicode mode matches ALL script word chars, so we only strip
    # ASCII-range punctuation by explicitly targeting non-word, non-space, non-hyphen-slash.
    # We narrow to ASCII punctuation by checking code-point range [!-~] minus alphanumeric:
    text = re.sub(r'[!-/:-@\[-`{-~]', ' ', text)  # strip ASCII punctuation block

    # Step 7: Strip leading/trailing hyphens/slashes
    text = RE_EDGE_PUNCT.sub('', text)

    # Step 8: Collapse whitespace
    text = RE_WHITESPACE.sub(' ', text).strip()

    return text


def normalize_address(addr_raw: str) -> str:
    """
    Normalize a business address string.

    Steps:
      1. Safe string conversion
      2. Lowercase
      3. Expand address abbreviations (word-boundary aware)
      4. Remove ASCII punctuation (preserve hyphens, slashes, non-ASCII scripts)
      5. Strip leading/trailing hyphens/slashes
      6. Collapse whitespace
    """
    text = _safe_str(addr_raw).lower()

    if not text:
        return ''

    # Step 3: Expand address abbreviations
    for pattern, replacement in ADDRESS_ABBR_PATTERNS:
        text = pattern.sub(replacement, text)

    # Step 4: Remove ASCII punctuation only (same approach as normalize_business_name)
    text = re.sub(r'[!-/:-@\[-`{-~]', ' ', text)

    # Step 5: Strip edge hyphens/slashes
    text = RE_EDGE_PUNCT.sub('', text)

    # Step 6: Collapse whitespace
    text = RE_WHITESPACE.sub(' ', text).strip()

    return text


def build_composite_key(name_norm: str, addr_norm: str) -> str:
    """
    Concatenate normalized name and address into a single blocking string.
    Used downstream for TF-IDF and MinHash blocking.
    """
    parts = [p for p in [name_norm, addr_norm] if p]
    return ' '.join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Chunk-Based Preprocessor
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_file(
    input_path: str,
    output_path: str,
    chunk_size: int = 100_000,
) -> None:
    """
    Read a source TSV in chunks, apply normalization, write preprocessed TSV.

    Parameters
    ----------
    input_path  : path to raw *.tsv
    output_path : path for output preprocessed_*.tsv
    chunk_size  : number of rows per chunk (default 100,000)
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows   = 0
    chunk_num    = 0
    t_start      = time.time()

    log.info(f"Processing: {input_path.name}  ->  {output_path.name}")
    log.info(f"Chunk size: {chunk_size:,} rows")

    reader = pd.read_csv(
        input_path,
        sep='\t',
        encoding='utf-8',
        chunksize=chunk_size,
        dtype=str,          # read everything as string — no silent NaN coercion
        keep_default_na=False,  # don't auto-convert 'nan', 'N/A', '' to NaN
        na_values=[''],     # only true empty cells become NaN
    )

    first_chunk = True

    for chunk in reader:
        chunk_num += 1
        t_chunk = time.time()

        # ── Validate expected columns ──────────────────────────────────────
        expected_cols = {'entity_id', 'business_name', 'business_address', 'country'}
        missing = expected_cols - set(chunk.columns)
        if missing:
            raise ValueError(f"Missing columns in {input_path.name}: {missing}")

        # ── Fill true NaN (empty cells) with empty string ──────────────────
        chunk['business_name']    = chunk['business_name'].fillna('')
        chunk['business_address'] = chunk['business_address'].fillna('')

        # ── Apply normalization ────────────────────────────────────────────
        chunk['name_norm'] = chunk['business_name'].map(normalize_business_name)
        chunk['addr_norm'] = chunk['business_address'].map(normalize_address)
        chunk['composite_key'] = [
            build_composite_key(n, a)
            for n, a in zip(chunk['name_norm'], chunk['addr_norm'])
        ]

        # ── Write (header only on first chunk) ────────────────────────────
        chunk.to_csv(
            output_path,
            sep='\t',
            index=False,
            mode='w' if first_chunk else 'a',
            header=first_chunk,
            encoding='utf-8',
        )
        first_chunk = False

        total_rows += len(chunk)
        elapsed    = time.time() - t_start
        throughput = total_rows / elapsed if elapsed > 0 else 0

        log.info(
            f"  Chunk {chunk_num:>4d} | rows {total_rows:>10,} | "
            f"{time.time() - t_chunk:.2f}s | "
            f"throughput {throughput:,.0f} rows/s"
        )

    total_elapsed = time.time() - t_start
    log.info(
        f"Done: {output_path.name} — "
        f"{total_rows:,} rows in {total_elapsed:.1f}s "
        f"({total_rows/total_elapsed:,.0f} rows/s)\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Preprocess source TSVs for entity resolution."
    )
    parser.add_argument(
        '--data-dir', default='dataset',
        help="Root directory containing train/ and test/ subdirs (default: dataset)"
    )
    parser.add_argument(
        '--out-dir', default='dataset/preprocessed',
        help="Output directory for preprocessed TSVs (default: dataset/preprocessed)"
    )
    parser.add_argument(
        '--chunk-size', type=int, default=100_000,
        help="Rows per chunk (default: 100,000)"
    )
    parser.add_argument(
        '--splits', nargs='+', default=['train', 'test'],
        choices=['train', 'test'],
        help="Which splits to preprocess (default: both train and test)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)

    # File mapping: split → list of (input_filename, output_filename)
    file_map = {
        'train': [
            ('train_source1.tsv', 'preprocessed_train_source1.tsv'),
            ('train_source2.tsv', 'preprocessed_train_source2.tsv'),
            ('train_source3.tsv', 'preprocessed_train_source3.tsv'),
        ],
        'test': [
            ('test_source1.tsv', 'preprocessed_test_source1.tsv'),
            ('test_source2.tsv', 'preprocessed_test_source2.tsv'),
            ('test_source3.tsv', 'preprocessed_test_source3.tsv'),
        ],
    }

    total_start = time.time()

    for split in args.splits:
        split_in_dir  = data_dir / split
        split_out_dir = out_dir  / split

        log.info(f"{'='*60}")
        log.info(f"Processing split: {split.upper()}")
        log.info(f"{'='*60}")

        for in_fname, out_fname in file_map[split]:
            input_path  = split_in_dir  / in_fname
            output_path = split_out_dir / out_fname

            if not input_path.exists():
                log.warning(f"File not found, skipping: {input_path}")
                continue

            preprocess_file(
                input_path=str(input_path),
                output_path=str(output_path),
                chunk_size=args.chunk_size,
            )

    total_elapsed = time.time() - total_start
    log.info(f"All preprocessing complete in {total_elapsed/60:.1f} minutes.")
    log.info(f"Output written to: {out_dir}")


if __name__ == '__main__':
    main()
