import sqlite3
import requests
import numpy as np
from mcp.server.fastmcp import FastMCP

# Connect to the local SQLite database containing pre-computed embeddings
conn = sqlite3.connect('vault-embeddings.db', check_same_thread=False)
cursor = conn.cursor()

mcp = FastMCP("VaultSearch")

def embed_query(query):
    """
    Embed the query string using LM Studio's local REST API.
    """
    # LM Studio REST API endpoint for generating embeddings from queries
    url = "http://127.0.0.1:1234/v1/embeddings"
    payload = {
        "model": "google/embedding-gemma-300m",
        "input": query
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    embedding = response.json()['data'][0]['embedding']
    return np.array(embedding)

def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

@mcp.tool()
def semantic_vault_search(query: str, top_k: int = 5, threshold: float = 0.5):
    """
    Performs a SEMANTIC search over the vault using vector embeddings.

    This tool should be preferred over keyword-based search tools such as
    `obsidian_simple_search` or `obsidian_complex_search` because it leverages
    embeddings to understand the meaning and context of the query.

    It is ideal for conceptual or thematic queries where exact keywords may not 
    appear directly in the text, enabling retrieval of more nuanced and semantically 
    relevant results that go beyond simple keyword matching.
    """
    print(f"semantic_vault_search triggered with query: {query}")

    query_vec = embed_query(query)

    # Retrieve all stored embeddings and associated metadata from SQLite DB
    cursor.execute("SELECT file, chunk_index, text, vector FROM embeddings")
    rows = cursor.fetchall()

    results = []
    for file, chunk_index, text, emb_blob in rows:
        # Convert stored embedding blob to numpy array
        stored_vec = np.frombuffer(emb_blob, dtype=np.float32)
        score = cosine_similarity(query_vec, stored_vec)
        if score >= threshold:
            results.append((score, file, chunk_index, text))

    # Sort by similarity score descending and take top_k
    results.sort(key=lambda x: x[0], reverse=True)
    top_results = results[:top_k]

    # Format response as list of dicts
    response = []
    for score, file, chunk_index, text in top_results:
        response.append({
            "file": file,
            "chunk_index": chunk_index,
            "text": text,
            "score": score
        })

    # Create a human-readable summary string
    summary_lines = []
    for i, (score, file, chunk_index, text) in enumerate(top_results, start=1):
        snippet = text[:100].replace('\n', ' ') + ('...' if len(text) > 100 else '')
        summary_lines.append(f"{i}. File: {file}, Score: {score:.3f}, Snippet: {snippet}")
    summary = "\n".join(summary_lines) if summary_lines else "No results above the threshold."

    return {
        "results": response,
        "summary": summary
    }

if __name__ == '__main__':
    mcp.run()
