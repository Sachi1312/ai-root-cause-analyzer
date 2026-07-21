from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import faiss
import numpy as np
import json
import os
import glob

load_dotenv()

# ──────────────────────────────────────────────
# FAISS RAG ENGINE
# FAISS = Facebook AI Similarity Search
# Runs 100% locally — no cloud, no API calls
# World's fastest vector similarity search
#
# How it works:
# 1. Past incidents stored as vectors in FAISS index
# 2. New log summary converted to vector
# 3. FAISS finds top-K most similar past incidents
# 4. Retrieved incidents become context for LLM
# ──────────────────────────────────────────────

embedder = SentenceTransformer("all-MiniLM-L6-v2")

DIMENSION = 384
INDEX_PATH = "faiss_index/incidents.index"
META_PATH = "faiss_index/metadata.json"


def load_or_create_index():
    """Load existing FAISS index or create a new one."""
    os.makedirs("faiss_index", exist_ok=True)

    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r") as f:
            metadata = json.load(f)
        print(f"Loaded FAISS index with {index.ntotal} incidents")
    else:
        # Why IndexFlatIP? Inner product similarity
        # For normalized vectors, this equals cosine similarity
        index = faiss.IndexFlatIP(DIMENSION)
        metadata = []
        print("Created new FAISS index")

    return index, metadata


def save_index(index, metadata):
    """Save FAISS index and metadata to disk."""
    os.makedirs("faiss_index", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)


def add_incident(incident: dict):
    """
    Add a past incident to the FAISS index.
    incident = {
        "id": "INC-2847",
        "title": "DB connection pool exhaustion",
        "description": "...",
        "root_cause": "...",
        "resolution": "...",
        "service": "payment-service",
        "severity": "P1",
        "resolved_in": "23 min"
    }
    """
    index, metadata = load_or_create_index()

    # Create searchable text from incident
    text = f"{incident['title']} {incident.get('description', '')} {incident.get('root_cause', '')}"

    # Embed and normalize for cosine similarity
    embedding = embedder.encode([text])[0]
    embedding = embedding / np.linalg.norm(embedding)
    embedding = embedding.reshape(1, -1).astype(np.float32)

    # Add to FAISS
    index.add(embedding)
    metadata.append(incident)

    save_index(index, metadata)
    print(f"Added incident {incident['id']} to FAISS index")


def search_similar(query_text: str, top_k: int = 3) -> list:
    """
    Find most similar past incidents to current log summary.
    Returns top_k results with similarity scores.
    """
    index, metadata = load_or_create_index()

    if index.ntotal == 0:
        return []

    # Embed query
    embedding = embedder.encode([query_text])[0]
    embedding = embedding / np.linalg.norm(embedding)
    embedding = embedding.reshape(1, -1).astype(np.float32)

    # Search FAISS
    k = min(top_k, index.ntotal)
    scores, indices = index.search(embedding, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0 and idx < len(metadata):
            result = metadata[idx].copy()
            result["similarity"] = round(float(score) * 100, 1)
            results.append(result)

    return results


def seed_sample_incidents():
    """
    Seed the FAISS index with realistic sample incidents.
    This gives the RAG something to search from the start.
    """
    index, metadata = load_or_create_index()

    if index.ntotal > 0:
        print(f"Index already has {index.ntotal} incidents, skipping seed")
        return

    sample_incidents = [
        {
            "id": "INC-2847",
            "title": "Database connection pool exhaustion in payment service",
            "description": "NullPointerException prevented connections from being released causing pool exhaustion and cascade timeouts",
            "root_cause": "Unhandled null reference in PaymentProcessor caused DB connections to not be released back to pool",
            "resolution": "Added null check, increased pool size from 10 to 25, added circuit breaker",
            "service": "payment-service",
            "severity": "P1",
            "resolved_in": "23 min",
            "date": "2026-03-12"
        },
        {
            "id": "INC-2634",
            "title": "API timeout cascade across order and notification services",
            "description": "Upstream payment service slowdown caused timeout cascade to order and notification services",
            "root_cause": "Database query without index caused full table scan, response time exceeded 30s timeout",
            "resolution": "Added database index on transaction_id column, increased timeout to 60s, added retry logic",
            "service": "order-service",
            "severity": "P2",
            "resolved_in": "41 min",
            "date": "2026-01-08"
        },
        {
            "id": "INC-2401",
            "title": "NullPointerException in payment processing flow",
            "description": "Missing null check in payment gateway response handler caused service crash",
            "root_cause": "Third-party payment gateway returned unexpected null field in response object",
            "resolution": "Added defensive null checks throughout gateway response handler, added input validation",
            "service": "payment-service",
            "severity": "P1",
            "resolved_in": "15 min",
            "date": "2025-11-19"
        },
        {
            "id": "INC-2198",
            "title": "Memory leak in authentication service causing OOM",
            "description": "JWT token objects not being garbage collected causing heap to fill over 6 hours",
            "root_cause": "Static cache holding JWT references prevented garbage collection of expired tokens",
            "resolution": "Changed static cache to WeakHashMap, added TTL-based eviction policy",
            "service": "auth-service",
            "severity": "P2",
            "resolved_in": "2 hours",
            "date": "2025-09-04"
        },
        {
            "id": "INC-1987",
            "title": "Redis cache connection pool exhaustion",
            "description": "Cache connection pool ran out during traffic spike causing 503 errors across all services",
            "root_cause": "Missing connection.close() calls in cache client wrapper class",
            "resolution": "Fixed connection leak, switched to connection pool with automatic cleanup",
            "service": "cache-service",
            "severity": "P1",
            "resolved_in": "35 min",
            "date": "2025-07-22"
        },
        {
            "id": "INC-1756",
            "title": "CPU spike in ML inference service causing latency",
            "description": "Model inference taking 10x longer than expected under load causing request queue buildup",
            "root_cause": "Model loaded fresh for each request instead of being cached in memory",
            "resolution": "Implemented model caching on service startup, added request batching",
            "service": "ml-service",
            "severity": "P3",
            "resolved_in": "1 hour",
            "date": "2025-05-11"
        },
        {
            "id": "INC-1534",
            "title": "Disk space exhaustion on logging service",
            "description": "Log rotation not configured causing disk to fill and service crashes",
            "root_cause": "Log rotation policy was accidentally removed during infrastructure migration",
            "resolution": "Re-enabled log rotation with 7-day retention, added disk space alerting",
            "service": "logging-service",
            "severity": "P2",
            "resolved_in": "20 min",
            "date": "2025-03-07"
        },
        {
            "id": "INC-1312",
            "title": "HTTP 503 cascade from overloaded API gateway",
            "description": "API gateway rate limiter misconfigured after deployment causing legitimate traffic to be blocked",
            "root_cause": "Rate limit config had wrong units — set to 100/hour instead of 100/second",
            "resolution": "Fixed rate limit config, rolled back deployment, implemented config validation",
            "service": "api-gateway",
            "severity": "P1",
            "resolved_in": "12 min",
            "date": "2025-01-18"
        }
    ]

    for incident in sample_incidents:
        add_incident(incident)

    print(f"Seeded {len(sample_incidents)} sample incidents into FAISS index")