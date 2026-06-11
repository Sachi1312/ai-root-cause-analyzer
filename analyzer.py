from log_parser import parse_logs, detect_anomalies, score_severity, extract_service_graph
from rag_engine import search_similar, seed_sample_incidents
from dotenv import load_dotenv
import requests
import os
import json

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "openrouter/free"


# ──────────────────────────────────────────────
# BART SUMMARISATION
# Why compress logs before sending to LLM?
# 1. Logs can be 50,000+ characters
# 2. LLMs have context limits (~8000 tokens)
# 3. BART is fine-tuned for summarisation
#    better than asking a general LLM to summarise
#
# We use a simple extractive approach if BART
# is too slow — picks most important lines
# ──────────────────────────────────────────────

def compress_logs(raw_text: str, parsed: dict, max_chars: int = 2000) -> str:
    """
    Compress logs to fit in LLM context window.
    Priority: ERROR > WARN > stack traces > other
    """
    if len(raw_text) <= max_chars:
        return raw_text

    important_lines = []

    # Always include error lines first
    for item in parsed["error_lines"][:10]:
        important_lines.append(item["text"])

    # Include warn lines
    for item in parsed["warn_lines"][:5]:
        important_lines.append(item["text"])

    # Include stack traces
    for trace in parsed["stack_traces"][:5]:
        important_lines.append(trace)

    compressed = "\n".join(important_lines)

    # If still too long, truncate
    if len(compressed) > max_chars:
        compressed = compressed[:max_chars] + "\n...[truncated]"

    return compressed


def call_llm(prompt: str, system: str = "") -> str:
    """Call OpenRouter API."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": MODEL,
            "messages": messages,
            "max_tokens": 1500,
            "temperature": 0.2
        },
        timeout=60
    )

    result = response.json()
    if "choices" not in result:
        raise Exception(f"LLM error: {result}")
    return result["choices"][0]["message"]["content"]


def analyze_root_cause(compressed_logs: str, similar_incidents: list, parsed: dict) -> dict:
    """Ask LLM to identify root cause with RAG context."""

    similar_context = ""
    if similar_incidents:
        similar_context = "\n\nSIMILAR PAST INCIDENTS (from RAG):\n"
        for inc in similar_incidents:
            similar_context += f"""
- {inc['id']} ({inc['similarity']}% match): {inc['title']}
  Root cause: {inc['root_cause']}
  Resolution: {inc['resolution']}
  Resolved in: {inc['resolved_in']}
"""

    prompt = f"""Analyze these server logs and identify the root cause of the failure.

LOGS:
{compressed_logs}

DETECTED PATTERNS:
- Errors: {parsed['error_count']}
- Services affected: {', '.join(parsed['services']) or 'unknown'}
- Exceptions: {', '.join(parsed['exceptions']) or 'none'}
{similar_context}

Return ONLY valid JSON in this exact format:
{{
    "root_cause": "Clear one paragraph explanation of the root cause",
    "confidence": 85,
    "affected_component": "specific service or component name",
    "failure_type": "one of: connection_pool, memory_leak, null_pointer, timeout, oom, cascade, config, unknown",
    "fixes": [
        {{"fix": "Specific actionable fix", "confidence": 90, "priority": "immediate"}},
        {{"fix": "Second fix", "confidence": 80, "priority": "short_term"}},
        {{"fix": "Third fix", "confidence": 70, "priority": "long_term"}}
    ],
    "runbook": [
        "Step 1: specific action",
        "Step 2: specific action",
        "Step 3: specific action",
        "Step 4: verify recovery"
    ]
}}"""

    raw = call_llm(prompt, system="You are a senior SRE engineer. Analyze logs and return only valid JSON.")

    # Clean JSON
    raw = raw.strip()
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(raw)
    except:
        return {
            "root_cause": raw,
            "confidence": 70,
            "affected_component": "unknown",
            "failure_type": "unknown",
            "fixes": [],
            "runbook": []
        }


def generate_report(company: str, analysis: dict, parsed: dict,
                    severity: dict, anomalies: dict) -> str:
    """Generate a Slack/PagerDuty ready incident report."""

    fixes_text = "\n".join([f"  {i+1}. {f['fix']}" for i, f in enumerate(analysis.get("fixes", []))])
    runbook_text = "\n".join([f"  {step}" for step in analysis.get("runbook", [])])
    anomaly_text = "\n".join([f"  • {a['icon']} {a['description']}" for a in anomalies.get("anomalies", [])])

    report = f"""## Incident Report

**Severity:** {severity['priority']} — {severity['label']}
**Services Affected:** {', '.join(parsed['services']) or 'Unknown'}
**Confidence:** {analysis.get('confidence', 0)}%

### Root Cause
{analysis.get('root_cause', 'Analysis pending')}

### Anomalies Detected
{anomaly_text or '  None detected'}

### Suggested Fixes
{fixes_text or '  No fixes generated'}

### Runbook
{runbook_text or '  No runbook generated'}

### Stats
- Total log lines: {parsed['total_lines']}
- Error lines: {parsed['error_count']}
- Warnings: {parsed['warn_count']}
- Exceptions: {', '.join(parsed['exceptions']) or 'None'}
"""
    return report


def run_full_analysis(raw_logs: str) -> dict:
    """
    Main pipeline — runs all analysis steps.
    Returns complete result dict for frontend.
    """
    print("\n[Analyzer] Starting full analysis pipeline...")

    # Step 1: Parse logs
    print("[1/6] Parsing logs...")
    parsed = parse_logs(raw_logs)
    print(f"      → {parsed['error_count']} errors, {len(parsed['services'])} services")

    # Step 2: Detect anomalies
    print("[2/6] Detecting anomalies...")
    anomalies = detect_anomalies(parsed, raw_logs)
    print(f"      → {anomalies['count']} anomalies found")

    # Step 3: Score severity
    print("[3/6] Scoring severity...")
    severity = score_severity(parsed, anomalies)
    print(f"      → {severity['priority']} ({severity['label']})")

    # Step 4: Compress logs
    print("[4/6] Compressing logs...")
    compressed = compress_logs(raw_logs, parsed)
    print(f"      → {len(raw_logs)} → {len(compressed)} chars")

    # Step 5: FAISS RAG search
    print("[5/6] Searching similar incidents (FAISS)...")
    similar = search_similar(compressed, top_k=3)
    print(f"      → {len(similar)} similar incidents found")

    # Step 6: LLM analysis
    print("[6/6] Running LLM analysis...")
    analysis = analyze_root_cause(compressed, similar, parsed)
    print(f"      → Root cause identified with {analysis.get('confidence', 0)}% confidence")

    # Extract service graph
    graph = extract_service_graph(parsed, raw_logs)

    # Generate report
    report = generate_report("", analysis, parsed, severity, anomalies)

    return {
        "parsed": {
            "total_lines": parsed["total_lines"],
            "error_count": parsed["error_count"],
            "warn_count": parsed["warn_count"],
            "services": parsed["services"],
            "exceptions": parsed["exceptions"],
            "keywords": parsed["keywords"]
        },
        "anomalies": anomalies["anomalies"],
        "severity": severity,
        "similar_incidents": similar,
        "analysis": analysis,
        "service_graph": graph,
        "report": report,
        "compressed_log_size": len(compressed)
    }


# Seed FAISS on import
seed_sample_incidents()