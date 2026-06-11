import re
from datetime import datetime
from collections import defaultdict


# ──────────────────────────────────────────────
# LOG PARSER
# Converts raw unstructured log text into
# structured data we can analyse programmatically.
#
# Why parse before sending to AI?
# 1. AI is expensive — structured data is cheaper
# 2. Regex is faster and more reliable for patterns
# 3. We can compute statistics AI cannot
# ──────────────────────────────────────────────

# Common log patterns
PATTERNS = {
    "timestamp": r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?',
    "time_only": r'\d{2}:\d{2}:\d{2}(?:\.\d+)?',
    "log_level": r'\b(ERROR|WARN|WARNING|INFO|DEBUG|FATAL|CRITICAL|SEVERE)\b',
    "exception": r'([A-Za-z]+(?:Exception|Error|Fault|Failure))[:\s]',
    "service": r'([a-z][a-z0-9]*(?:-[a-z0-9]+)*)-service',
    "http_status": r'\b([45]\d{2})\b',
    "memory": r'(\d+(?:\.\d+)?)\s*(?:MB|GB|KB)\s*(?:heap|memory|RAM)',
    "line_number": r'(?:line|at)\s+(\d+)',
    "ip_address": r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b',
    "duration": r'(\d+(?:\.\d+)?)\s*(?:ms|seconds|s)\b',
    "stack_trace": r'^\s+at\s+[\w.$]+\(',
    "oom": r'OutOfMemoryError|OOM|heap space|GC overhead',
    "timeout": r'(?:timeout|timed out|connection refused)',
    "null_pointer": r'NullPointerException|NPE|null reference',
    "db_error": r'(?:connection pool|database|DB|SQL|jdbc)',
}


def parse_logs(raw_text: str) -> dict:
    """
    Parse raw log text into structured data.
    Returns a comprehensive analysis dict.
    """
    lines = raw_text.strip().split('\n')
    
    parsed = {
        "total_lines": len(lines),
        "error_lines": [],
        "warn_lines": [],
        "info_lines": [],
        "exceptions": [],
        "services": set(),
        "http_errors": [],
        "timestamps": [],
        "stack_traces": [],
        "keywords": defaultdict(int),
        "error_count": 0,
        "warn_count": 0,
        "info_count": 0,
    }

    for i, line in enumerate(lines):
        # Extract log level
        level_match = re.search(PATTERNS["log_level"], line, re.IGNORECASE)
        level = level_match.group(1).upper() if level_match else "UNKNOWN"

        # Categorize by level
        if level in ["ERROR", "FATAL", "CRITICAL", "SEVERE"]:
            parsed["error_lines"].append({"line_num": i+1, "text": line.strip()})
            parsed["error_count"] += 1
        elif level in ["WARN", "WARNING"]:
            parsed["warn_lines"].append({"line_num": i+1, "text": line.strip()})
            parsed["warn_count"] += 1
        elif level == "INFO":
            parsed["info_count"] += 1

        # Extract timestamps
        ts = re.search(PATTERNS["timestamp"], line)
        if not ts:
            ts = re.search(PATTERNS["time_only"], line)
        if ts:
            parsed["timestamps"].append(ts.group())

        # Extract exceptions
        exc = re.search(PATTERNS["exception"], line)
        if exc:
            parsed["exceptions"].append(exc.group(1))

        # Extract service names
        svc = re.findall(PATTERNS["service"], line)
        for s in svc:
            parsed["services"].add(s + "-service")

        # Extract HTTP errors
        http = re.search(PATTERNS["http_status"], line)
        if http:
            parsed["http_errors"].append(http.group(1))

        # Detect stack traces
        if re.match(PATTERNS["stack_trace"], line):
            parsed["stack_traces"].append(line.strip())

        # Count keywords
        for keyword in ["timeout", "null", "memory", "connection", "database",
                        "refused", "failed", "crash", "exception", "error"]:
            if keyword.lower() in line.lower():
                parsed["keywords"][keyword] += 1

    # Convert set to list for JSON serialization
    parsed["services"] = list(parsed["services"])
    parsed["exceptions"] = list(set(parsed["exceptions"]))
    parsed["keywords"] = dict(parsed["keywords"])

    return parsed


def detect_anomalies(parsed: dict, raw_text: str) -> dict:
    """
    Detect anomaly patterns in parsed logs.
    These are things AI might miss but regex catches reliably.

    Patterns detected:
    - Error spike: many errors in short time window
    - Cascade failure: multiple services failing in sequence
    - Memory leak: growing memory warnings
    - OOM: out of memory errors
    - Connection pool exhaustion: pool full messages
    - Timeout cascade: timeouts spreading across services
    """
    anomalies = []
    raw_lower = raw_text.lower()

    # Error spike detection
    if parsed["error_count"] > 5:
        anomalies.append({
            "type": "error_spike",
            "severity": "high",
            "description": f"{parsed['error_count']} errors detected — possible error spike",
            "icon": "⚡"
        })

    # Cascade failure detection
    if len(parsed["services"]) >= 2 and parsed["error_count"] >= 2:
        anomalies.append({
            "type": "cascade_failure",
            "severity": "critical",
            "description": f"Multiple services affected: {', '.join(parsed['services'][:4])}",
            "icon": "🔥"
        })

    # OOM detection
    if re.search(PATTERNS["oom"], raw_text, re.IGNORECASE):
        anomalies.append({
            "type": "out_of_memory",
            "severity": "critical",
            "description": "OutOfMemoryError or heap exhaustion detected",
            "icon": "💾"
        })

    # Memory leak detection
    if parsed["keywords"].get("memory", 0) >= 2:
        anomalies.append({
            "type": "memory_pressure",
            "severity": "medium",
            "description": "Multiple memory-related warnings — possible memory leak",
            "icon": "📈"
        })

    # Connection pool exhaustion
    if re.search(r'connection pool', raw_lower) and re.search(r'exhaust|full|maximum', raw_lower):
        anomalies.append({
            "type": "connection_pool_exhaustion",
            "severity": "high",
            "description": "Database connection pool exhausted",
            "icon": "🔌"
        })

    # Timeout cascade
    timeout_count = len(re.findall(PATTERNS["timeout"], raw_text, re.IGNORECASE))
    if timeout_count >= 2:
        anomalies.append({
            "type": "timeout_cascade",
            "severity": "high",
            "description": f"Multiple timeouts detected ({timeout_count}) — possible cascade",
            "icon": "⏱️"
        })

    # NullPointerException
    if re.search(PATTERNS["null_pointer"], raw_text, re.IGNORECASE):
        anomalies.append({
            "type": "null_pointer",
            "severity": "high",
            "description": "NullPointerException detected — unhandled null reference",
            "icon": "🚫"
        })

    return {"anomalies": anomalies, "count": len(anomalies)}


def score_severity(parsed: dict, anomalies: dict) -> dict:
    """
    Auto-score incident severity P1-P5 based on patterns.

    Scoring system:
    P1 - Critical: cascade failure, OOM, multiple services down
    P2 - High: connection pool, timeout cascade, many errors
    P3 - Medium: single service errors, memory pressure
    P4 - Low: warnings only, single errors
    P5 - Info: informational messages only
    """
    score = 0

    # Anomaly-based scoring
    for anomaly in anomalies["anomalies"]:
        if anomaly["severity"] == "critical":
            score += 40
        elif anomaly["severity"] == "high":
            score += 25
        elif anomaly["severity"] == "medium":
            score += 15

    # Error count scoring
    score += min(parsed["error_count"] * 3, 30)

    # Multi-service impact
    if len(parsed["services"]) >= 3:
        score += 20
    elif len(parsed["services"]) >= 2:
        score += 10

    # Exception scoring
    if parsed["exceptions"]:
        score += 10

    # HTTP 5xx errors
    if parsed["http_errors"]:
        score += 10

    # Convert score to P1-P5
    if score >= 80:
        priority = "P1"
        label = "Critical"
        color = "#E24B4A"
    elif score >= 55:
        priority = "P2"
        label = "High"
        color = "#BA7517"
    elif score >= 30:
        priority = "P3"
        label = "Medium"
        color = "#639922"
    elif score >= 10:
        priority = "P4"
        label = "Low"
        color = "#378ADD"
    else:
        priority = "P5"
        label = "Info"
        color = "#888780"

    return {
        "priority": priority,
        "label": label,
        "color": color,
        "score": score,
        "factors": {
            "error_count": parsed["error_count"],
            "services_affected": len(parsed["services"]),
            "anomaly_count": anomalies["count"],
            "has_exceptions": len(parsed["exceptions"]) > 0
        }
    }


def extract_service_graph(parsed: dict, raw_text: str) -> dict:
    """
    Extract service dependency graph from logs.
    Identifies which service caused cascade failures.

    Returns nodes and edges for D3.js visualisation.
    """
    services = parsed["services"]
    nodes = []
    edges = []

    # If no services detected, add generic ones based on common patterns
    if not services:
        for pattern in ["api", "auth", "payment", "order", "notification", "db", "cache"]:
            if pattern in raw_text.lower():
                services.append(pattern + "-service")

    # Create nodes
    for i, svc in enumerate(services[:8]):
        # Determine if this service is a likely root cause
        is_root = any(
            svc.replace("-service", "") in exc.lower()
            for exc in parsed["exceptions"]
        ) or (i == 0 and parsed["error_count"] > 0)

        nodes.append({
            "id": svc,
            "label": svc,
            "is_root": is_root,
            "error_count": parsed["error_count"] if is_root else 1
        })

    # Create edges (simple cascade assumption)
    for i in range(len(nodes) - 1):
        edges.append({
            "source": nodes[i]["id"],
            "target": nodes[i + 1]["id"],
            "type": "cascade"
        })

    return {"nodes": nodes, "edges": edges}