# Obsidian Vault Index for Embedding LLM Model

---

## 📌 Intent

This project indexes the Markdown files in an Obsidian Vault, splits them into manageable text chunks, generates embeddings for each chunk using LM Studio with the `google/embedding-gemma-300m` model, and stores these embeddings and associated metadata in a SQLite database. The resulting database enables efficient semantic search and retrieval-augmented generation (RAG) workflows on your local notes.

---

## ⚙️ Actions

The script performs the following main actions:

- Traverses all Markdown (`.md`) files within your Obsidian Vault, recursively.
- Splits each file’s content into smaller text chunks for optimal embedding.
- Sends each chunk to the local LM Studio API server, using the `google/embedding-gemma-300m` embedding model, to generate vector representations.
- Stores each embedding, along with file path, chunk index, and original chunk text, in a SQLite database.
- Supports directory filtering with:
  - `--dir <subdir>`: Only index a specific subdirectory.
  - `--exclude <subdir>`: Exclude one or more subdirectories from indexing.

---

## 📥 Input

- **Vault Path:** Set within the script (`vault-index.py`) to point to your Obsidian Vault directory.
- **LM Studio API:** Requires LM Studio running locally with the embedding model loaded. The API should be accessible at `http://127.0.0.1:1234/v1/embeddings`.
- **Python Environment:** Python 3.x is required.
- **Dependencies:** The `requests` library must be installed. Requests is available on PyPI: `pip install requests`


---

## 📤 Output

- **SQLite Database File:** All embeddings and metadata are stored in `vault-embeddings.db` (default) in the same directory as the script, unless the path is changed in the script.
- **Database Schema:**  
  Table: `embeddings`
  - `id` (INTEGER PRIMARY KEY)
  - `file` (TEXT) – relative path to the Markdown file
  - `chunk_index` (INTEGER) – index of the chunk within the file
  - `text` (TEXT) – chunk content
  - `vector` (BLOB or TEXT) – embedding vector, stored as a **float32 BLOB** for efficiency (can be reconstructed later in Python using `array("f").frombytes(blob)`)

---

## 🧱 Framework

- **Language:** Python 3.x
- **Dependencies:**  
  - `requests` (install via pip)  
  - `sqlite3` (Python standard library)

**Install dependencies:**
```bash
source ~/python-venv/bin/activate
pip install requests
```

**Run the indexer:**
```bash
python vault-index.py [--dir <subdir>] [--exclude <subdir> ...]
```

---

## 📝 Examples

- **Index the entire vault, excluding bibtex and Readwise directories:**
    ```bash
    python vault-index.py --exclude "bibtex-to-markdown" --exclude "Readwise"
    ```
- **Index only the bibtex directory:**
    ```bash
    python vault-index.py --dir "bibtex-to-markdown"
    ```
- **Index only the Readwise directory:**
    ```bash
    python vault-index.py --dir "Readwise"
    ```

---

## Usage Notes

- For large vaults, run the script in batches using `--dir` or `--exclude` to avoid overloading memory or the embedding model.
- The database accumulates embeddings across multiple runs; re-running will add new embeddings for new or changed files/chunks.
- Quantized models in LM Studio are recommended for faster, more efficient local embedding generation.

---

## 🛠️ Troubleshooting

### 🟥 Problem: Script runs but no embeddings are stored.
✅ Solution: Ensure LM Studio is running on your machine, the embedding model is loaded, and the API endpoint (`http://127.0.0.1:1234/v1/embeddings`) is accessible.

### 🟥 Problem: `ModuleNotFoundError: requests`
✅ Solution: Install the requests library with `pip install requests`.

### 🟥 Problem: Out-of-memory errors or script slowdown.
✅ Solution: Use `--dir` and/or `--exclude` to process the vault in smaller batches. Consider reducing the chunk size in the script if needed.

---

For enhancements or bug reports, please open an issue in the [GitHub repository](https://github.com/sjelms/*).