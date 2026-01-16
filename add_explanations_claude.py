#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# =========================
# CONFIG
# =========================

MODEL_ID_DEFAULT = "claude-sonnet-4-5-20250929"
ANTHROPIC_VERSION = "2023-06-01"
API_URL = "https://api.anthropic.com/v1/messages"

# Store explanation at item["q"]["ex"]
EXPLANATION_FIELD_PATH = ("q", "ex")

# Speed levers
MAX_TOKENS = 140          # was 220
TEMPERATURE = 0.3
REQUEST_TIMEOUT = 120

DEFAULT_WORKERS = 60      # was 10
DEFAULT_WRITE_EVERY = 96  # write after N successful updates
DEFAULT_MAX_PER_FILE = 0

MAX_RETRIES = 6
BASE_BACKOFF_SEC = 1.0
MAX_BACKOFF_SEC = 30.0

# If your stems are huge, this matters a lot:
STRIP_HTML = True
MAX_STIMULUS_CHARS = 3500  # truncate long passages to keep latency down (0 disables)

SYSTEM_PROMPT = (
    "You are a precise AP-style tutor. "
    "Write clear, correct, concise explanations. "
    "Plain text only. No markdown. "
    "End by clearly stating why the correct choice is correct."
)

MATH_PROMPT_TEMPLATE = """You are given ONE multiple-choice math/STEM question.

Write a 3–7 sentence explanation showing the key reasoning or method.
Use math when needed. Be direct and AP-style.

Return ONLY the explanation text.

QUESTION_ID: {qid}
Subject: {subject}

STIMULUS:
{stem_text}

OPTIONS:
{options_block}

CORRECT_OPTION_ID: {answer_id}
"""

HUM_PROMPT_TEMPLATE = """You are given ONE multiple-choice humanities question.

Write a 3–6 sentence explanation justifying the correct answer
using the core concept or definition.

Return ONLY the explanation text.

QUESTION_ID: {qid}
Subject: {subject}

STIMULUS:
{stem_text}

OPTIONS:
{options_block}

CORRECT_OPTION_ID: {answer_id}
"""

# =========================
# HELPERS
# =========================

def eprint(*a):
    print(*a, file=sys.stderr)

def atomic_write_json(path: Path, data: Any):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    tmp.replace(path)

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def find_itemdata_files(root: Path):
    return sorted(root.rglob("itemData.json"))

def get_in(d, path):
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur

def set_in(d, path, val):
    cur = d
    for k in path[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[path[-1]] = val

def has_explanation(item):
    v = get_in(item, EXPLANATION_FIELD_PATH)
    return isinstance(v, str) and v.strip() != ""

def get_subject(item):
    tg = (item.get("q") or {}).get("tg") or []
    if not isinstance(tg, list):
        return ""
    for t in tg:
        if isinstance(t, str) and t.startswith("Subject:"):
            return t.replace("Subject:", "").strip()
    return ""

def classify_subject(subject: str):
    s = (subject or "").lower()
    for k in [
        "calculus","physics","chemistry","biology","statistics",
        "precalculus","environmental","computer science","math"
    ]:
        if k in s:
            return "MATH"
    return "HUM"

def build_options_block(options):
    out = []
    if isinstance(options, list):
        for o in options:
            if isinstance(o, dict):
                v = o.get("v")
                l = o.get("l")
                if v is None and l is None:
                    continue
                out.append(f"{v}: {l}")
    return "\n".join(out)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

def normalize_stimulus(stem_html: str) -> str:
    txt = stem_html or ""
    if STRIP_HTML:
        txt = _HTML_TAG_RE.sub(" ", txt)
        txt = _WS_RE.sub(" ", txt).strip()
    if MAX_STIMULUS_CHARS and len(txt) > MAX_STIMULUS_CHARS:
        txt = txt[:MAX_STIMULUS_CHARS].rstrip() + "…"
    return txt

@dataclass
class WorkItem:
    qid: str
    item: Dict[str, Any]
    subject: str
    kind: str

# =========================
# HTTP SESSION (connection pooling)
# =========================

_thread_local = {}

def get_session() -> requests.Session:
    # One Session per thread for safe connection pooling
    sess = _thread_local.get("session")
    if sess is None:
        sess = requests.Session()
        _thread_local["session"] = sess
    return sess

# =========================
# API CALL
# =========================

def call_claude(api_key: str, model: str, system_prompt: str, user_prompt: str) -> str:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }

    sess = get_session()
    r = sess.post(API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    if r.status_code >= 400:
        # bubble up full text so retry can decide
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")

    data = r.json()
    parts = []
    for c in data.get("content", []):
        if isinstance(c, dict) and c.get("type") == "text":
            parts.append(c.get("text", ""))
    return "\n".join(parts).strip()

def call_with_retry(api_key: str, model: str, system_prompt: str, user_prompt: str) -> str:
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return call_claude(api_key, model, system_prompt, user_prompt)
        except Exception as e:
            last_err = e
            if attempt == MAX_RETRIES - 1:
                raise
            sleep = min(MAX_BACKOFF_SEC, BASE_BACKOFF_SEC * (2 ** attempt))
            # jitter
            time.sleep(sleep * (0.7 + random.random() * 0.6))
    raise last_err  # unreachable

# =========================
# CORE
# =========================

def generate_one(api_key: str, model: str, w: WorkItem) -> Tuple[str, Optional[str]]:
    q = w.item.get("q") or {}
    stem = q.get("s")
    options = q.get("o")
    answer = q.get("a")

    if not stem or not answer or not options:
        return w.qid, None

    stem_text = normalize_stimulus(stem)
    options_block = build_options_block(options)
    if not options_block.strip():
        return w.qid, None

    prompt = (MATH_PROMPT_TEMPLATE if w.kind == "MATH" else HUM_PROMPT_TEMPLATE).format(
        qid=w.qid,
        subject=w.subject or "Unknown",
        stem_text=stem_text,
        options_block=options_block,
        answer_id=answer,
    )

    text = call_with_retry(api_key, model, SYSTEM_PROMPT, prompt)
    return w.qid, text

def process_itemdata_file(api_key: str, model: str, path: Path, workers: int, write_every: int, max_per_file: int):
    data = load_json(path)
    targets: List[WorkItem] = []

    for qid, item in data.items():
        if not isinstance(item, dict):
            continue
        if has_explanation(item):
            continue
        q = item.get("q") or {}
        if not q.get("o") or not q.get("a") or not q.get("s"):
            continue

        subject = get_subject(item)
        kind = classify_subject(subject)
        targets.append(WorkItem(qid, item, subject, kind))

        if max_per_file and len(targets) >= max_per_file:
            break

    if not targets:
        print(f"[OK] {path}")
        return

    print(f"[FILE] {path} -> {len(targets)}")

    updated = 0
    processed = 0
    last_write_at = 0

    # ONE executor per file (big speed win)
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_qid = {
            pool.submit(generate_one, api_key, model, w): w.qid
            for w in targets
        }

        for fut in cf.as_completed(future_to_qid):
            qid = future_to_qid[fut]
            processed += 1
            try:
                _, expl = fut.result()
            except Exception as e:
                # keep going; print occasionally
                if processed <= 10 or processed % 50 == 0:
                    eprint(f"  [ERR] {qid}: {e}")
                expl = None

            if expl:
                set_in(data[qid], EXPLANATION_FIELD_PATH, expl)
                updated += 1

            # Write periodically to avoid losing progress, but not too often
            if (updated - last_write_at) >= write_every:
                atomic_write_json(path, data)
                last_write_at = updated
                print(f"  [PROGRESS] processed {processed}/{len(targets)} | saved {updated}")

            if processed % 100 == 0:
                print(f"  [PROGRESS] processed {processed}/{len(targets)} | saved {updated}")

    # final write
    atomic_write_json(path, data)
    print(f"[DONE] {path} (+{updated})")

# =========================
# ENTRY
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.getcwd())
    ap.add_argument("--model", default=MODEL_ID_DEFAULT)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--write-every", type=int, default=DEFAULT_WRITE_EVERY)
    ap.add_argument("--max-per-file", type=int, default=DEFAULT_MAX_PER_FILE)
    args = ap.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set")

    files = find_itemdata_files(Path(args.root))
    print(f"Found {len(files)} itemData.json files")

    for f in files:
        process_itemdata_file(api_key, args.model, f, args.workers, args.write_every, args.max_per_file)

if __name__ == "__main__":
    main()
