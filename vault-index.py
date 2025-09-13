import os
import glob
import json
import sqlite3
import requests
import argparse
from array import array

# === CONFIGURATION ===
VAULT_PATH = "/Users/stephenelms/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault"  # Set your vault directory here
DB_PATH = "vault-embeddings.db"  # Path to the SQLite database
CHUNK_WORDS = 500  # Approximate words per chunk
EMBEDDING_API_URL = "http://127.0.0.1:1234/v1/embeddings"
EMBEDDING_MODEL = "google/embedding-gemma-300m"


def get_md_files(vault_path):
    """Recursively find all .md files in the vault."""
    md_files = []
    for root, dirs, files in os.walk(vault_path):
        for file in files:
            if file.endswith(".md"):
                md_files.append(os.path.join(root, file))
    return md_files


def filter_files(md_files, target_dir=None, exclude_dirs=None):
    """Filter md_files based on target_dir and exclude_dirs."""
    filtered_files = []
    exclude_dirs = exclude_dirs or []
    for file_path in md_files:
        rel_path = os.path.relpath(file_path, VAULT_PATH)
        if target_dir:
            if not rel_path.startswith(target_dir):
                continue
        if any(rel_path.startswith(excl) for excl in exclude_dirs):
            continue
        filtered_files.append(file_path)
    return filtered_files


def split_into_chunks(text, chunk_words=500):
    """Split text into chunks of approximately chunk_words words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_words):
        chunk = " ".join(words[i:i + chunk_words])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def get_embedding(text):
    """Send text to LM Studio embedding endpoint and return the vector."""
    payload = {
        "model": EMBEDDING_MODEL,
        "input": text,
    }
    try:
        resp = requests.post(EMBEDDING_API_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
        # Expecting: {"data": [{"embedding": [...] }], ...}
        embedding = data["data"][0]["embedding"]
        return embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None


def ensure_db_schema(conn):
    """Ensure the embeddings table exists."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY,
            file TEXT,
            chunk_index INTEGER,
            text TEXT,
            vector BLOB
        )"""
    )
    conn.commit()


def store_embedding(conn, file, chunk_index, text, vector):
    """Store embedding in the database.

    Vectors are stored as float32 BLOBs. To reconstruct, use:
        arr = array("f"); arr.frombytes(blob)
    """
    blob = array("f", vector).tobytes()
    conn.execute(
        "INSERT INTO embeddings (file, chunk_index, text, vector) VALUES (?, ?, ?, ?)",
        (file, chunk_index, text, blob),
    )
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Index Obsidian vault markdown files with embeddings.")
    parser.add_argument("--dir", type=str, default=None, help="Relative subdirectory under vault to index")
    parser.add_argument("--exclude", type=str, action="append", default=[], help="Relative subdirectories to exclude")
    args = parser.parse_args()

    print(f"Indexing Obsidian vault: {VAULT_PATH}")

    md_files = get_md_files(VAULT_PATH)
    if args.dir:
        print(f"Target directory to index: {args.dir}")
    if args.exclude:
        print(f"Excluding directories: {args.exclude}")

    filtered_files = filter_files(md_files, target_dir=args.dir, exclude_dirs=args.exclude)
    print(f"Found {len(filtered_files)} markdown files after filtering.")

    conn = sqlite3.connect(DB_PATH)
    ensure_db_schema(conn)

    total_chunks = 0
    for idx, file_path in enumerate(filtered_files):
        rel_file = os.path.relpath(file_path, VAULT_PATH)
        print(f"[{idx+1}/{len(filtered_files)}] Processing: {rel_file}")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = split_into_chunks(text, CHUNK_WORDS)
        for chunk_idx, chunk in enumerate(chunks):
            print(f"  - Chunk {chunk_idx+1}/{len(chunks)}", end="\r")
            vector = get_embedding(chunk)
            if vector is not None:
                store_embedding(conn, rel_file, chunk_idx, chunk, vector)
                total_chunks += 1
        print(f"  --> {len(chunks)} chunks indexed.")

    print(f"Indexing complete. {total_chunks} chunks embedded and stored in {DB_PATH}.")
    conn.close()


if __name__ == "__main__":
    main()