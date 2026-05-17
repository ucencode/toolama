#!/usr/bin/env python3

import os
import re
import json
import subprocess
import argparse
import time
from datetime import datetime
from pathlib import Path
from ollama import chat, ChatResponse


# ── constants ─────────────────────────────────────────────────────────────────

MODEL_KEYWORDS = ["llama3", "qwen3", "gemma", "mistral", "deepseek", "phi", "gpt-oss"]

LANG_INSTRUCTION = {
    "auto": "the same language as the source content",
    "ar": "العربية (Arabic)",
    "de": "Deutsch (German)",
    "en": "English",
    "es": "Español (Spanish)",
    "fi": "Suomi (Finnish)",
    "fr": "Français (French)",
    "hi": "हिन्दी (Hindi)",
    "id": "Bahasa Indonesia",
    "it": "Italiano (Italian)",
    "ja": "日本語 (Japanese)",
    "ko": "한국어 (Korean)",
    "nl": "Nederlands (Dutch)",
    "pl": "Polski (Polish)",
    "pt": "Português (Portuguese)",
    "ru": "Русский (Russian)",
    "sv": "Svenska (Swedish)",
    "th": "ภาษาไทย (Thai)",
    "tr": "Türkçe (Turkish)",
    "uk": "Українська (Ukrainian)",
    "vi": "Tiếng Việt (Vietnamese)",
    "zh": "简体中文 (Chinese)",
}

LANG_EXPERIMENTAL = {
    "ja", "ko", "it", "nl", "pl", "tr",
    "hi", "vi", "uk", "fi", "sv", "th",
}

MODES = ("plan", "full")


# ── system prompts ────────────────────────────────────────────────────────────

META_SYSTEM = """\
You are a curriculum metadata extractor.

Extract structured metadata from the curriculum text the user provides.
Return ONLY a valid JSON object with exactly these fields:

{
  "title":           "string  — e.g. 'Self-Study Plan: Calculus'",
  "course":          "string  — normalized course name, title case",
  "course_code":     "string  — if explicitly present (e.g. CS101), else empty string",
  "credits":         "integer — 0 if not found",
  "topics":          ["list of topic strings"],
  "outcomes":        ["list of learning outcome strings"],
  "topics_count":    "integer",
  "outcomes_count":  "integer",
  "estimated_weeks": "integer — roughly topics_count * 1.5, rounded",
  "tags":            ["2-4 lowercase subject area keywords"],
  "status":          "draft"
}

Rules:
- Normalize course name to title case
- title field must follow pattern: "Self-Study Plan: <Course Name>"
- course_code only if explicitly present, else empty string
- Raw JSON only — no explanation, no markdown fences, no preamble"""

META_USER = "Extract metadata from this curriculum:\n\n{raw}"

PLAN_SYSTEM = """\
You are a study plan architect for a self-directed learner.

Learner context:
- Works a full-time job; studies on weekday evenings (~1-2 hrs) and weekends (~4 hrs)
- Does NOT trust surface-level explanations — wants real conceptual understanding
- Attends formal classes but treats them only as a loose reference
- Goal: genuine mastery, not just passing exams

Output rules:
- Format: Markdown only
- Do NOT include YAML frontmatter — it is prepended separately
- Structure:
    1. Realistic weekly schedule with hours/week breakdown
    2. Phase-by-phase topic breakdown with estimated weeks per phase
    3. Per topic: what to focus on, what to skip, and a "you understand this when..." checkpoint
    4. Recommended free resources — specific titles only, no generic advice
    5. Weekly review ritual to consolidate learning
- At the end of EVERY phase, include this exact block:

> **Go Deeper** *(optional)*
> - <Specific book title + chapter, or named concept, or harder problem set>
> - <Second recommendation>
> - <Third recommendation, if relevant>
> *Pursue these only if you want to go beyond the phase scope.*

- Be direct and opinionated; skip hedging and filler
- Assume the learner is intelligent but time-constrained"""

PLAN_USER = "Respond in {lang_name}.\n\nCurriculum:\n\"\"\"\n{raw}\n\"\"\""

MATERIAL_SYSTEM = """\
You are a technical study material writer for a self-directed learner.

Learner context:
- Works a full-time job; limited study time — material must be dense, not padded
- Wants real understanding, not surface definitions
- Will use this material alongside a study plan, so assume topic order is intentional

Output rules:
- Format: Markdown only
- Do NOT include YAML frontmatter
- Write every topic using EXACTLY this template, in this order, with these exact headings:

---
## <Topic Name>

### Concept & Intuition
Explain the core idea. Build intuition first, then formalize.
Cover the "why this exists" before the "how it works".
Use analogies only where they genuinely clarify.

### Worked Examples
Minimum 2 worked examples, stepped through clearly.
Show reasoning at each step, not just the mechanics.

### Practice Problems
3-5 problems of increasing difficulty. No solutions.
Last problem must stretch beyond routine application.

### Common Misconceptions
2-3 specific wrong mental models people bring INTO this topic.
State the misconception explicitly, then correct it precisely.
No generic warnings — name the exact flawed assumption.

### Go Deeper *(optional — pursue only if curious)*
2-4 concrete recommendations for going beyond this topic.
Each entry must name: a specific book + chapter, a named theorem,
a specific paper, or a precisely described problem class.
No generic advice like "read more about X".
---

- Repeat this exact structure for every topic, in the order listed
- Be precise and direct; no filler, no repetition across topics
- Assume the learner is intelligent but time-constrained"""

MATERIAL_USER = """\
Respond in {lang_name}.

Write study material for every topic in this curriculum, in the order listed:

{topics}

Curriculum context (for depth calibration):
\"\"\"
{raw}
\"\"\""""

TOPIC_EXTRACT_SYSTEM = """\
You are a topic list extractor.

Given a study plan in Markdown, extract a flat, ordered list of every distinct topic
and subtopic the plan covers — including any the plan adds beyond the raw curriculum.

Return ONLY a valid JSON array of strings, one entry per topic, in the order they appear.
No explanations, no markdown fences, no nesting."""

TOPIC_EXTRACT_USER = "Extract the topic list from this study plan:\n\n{plan}"


# ── ollama model discovery ────────────────────────────────────────────────────

def list_models() -> list[str]:
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[error] 'ollama list' failed: {result.stderr.strip()}")
        exit(1)
    models = []
    for line in result.stdout.strip().splitlines()[1:]:
        name = line.split()[0]
        if any(kw in name for kw in MODEL_KEYWORDS):
            models.append(name)
    return models


def eject_model(model: str):
    print(f"[ollama] ejecting {model}...", end=" ", flush=True)
    result = subprocess.run(["ollama", "stop", model], capture_output=True, text=True)
    print("done" if result.returncode == 0 else f"warn: {result.stderr.strip()}")


# ── interactive prompts ───────────────────────────────────────────────────────

def ask_model(models: list[str]) -> str:
    print("\nSelect model:")
    for i, m in enumerate(models, 1):
        print(f"  {i}. {m}")
    print("[default: 1]")
    choice = input(">>> ").strip() or "1"
    try:
        return models[int(choice) - 1]
    except (ValueError, IndexError):
        print(f"[warn] invalid choice, using {models[0]}")
        return models[0]


def ask_language() -> str:
    codes = list(LANG_INSTRUCTION.keys())
    print(f"\nOutput language? ({' / '.join(codes)}) [default: auto]")
    lang = input(">>> ").strip().lower() or "auto"
    if lang not in LANG_INSTRUCTION:
        print(f"[warn] unknown language '{lang}', using auto")
        return "auto"
    if lang in LANG_EXPERIMENTAL:
        print(f"[warn] '{LANG_INSTRUCTION[lang]}' quality depends on model proficiency. Results may vary.")
    return lang


def ask_mode() -> str:
    print("""
Mode?
  1. plan   — study plan only
  2. full   — study plan + study material per topic
[default: 1]""")
    choice = input(">>> ").strip() or "1"
    return {"1": "plan", "2": "full"}.get(choice, "plan")


def ask_file() -> str:
    from glob import glob
    txt_files = sorted(glob("*.txt"))
    if txt_files:
        print("\nCurriculum file? (found in current directory:)")
        for i, f in enumerate(txt_files, 1):
            print(f"  {i}. {f}")
        print("Or enter a path manually.")
    else:
        print("\nCurriculum file? (path to .txt file)")
    choice = input(">>> ").strip()
    if choice.isdigit() and txt_files:
        idx = int(choice) - 1
        if 0 <= idx < len(txt_files):
            return txt_files[idx]
    return choice


# Sanitize JSON by removing markdown code fences if present, and trimming whitespace.
def _sanitize_json(raw: str) -> str:
    """Strip common model artifacts that break json.loads."""
    # strip markdown fences
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE)
    # strip // line comments
    raw = re.sub(r"//[^\n]*", "", raw)
    # strip # line comments (only when not inside a string — best-effort)
    raw = re.sub(r"(?<![\"'\w])#[^\n]*", "", raw)
    # strip trailing commas before ] or }
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    return raw.strip()


# ── call 1: frontmatter (json, no stream) ─────────────────────────────────────

def generate_frontmatter(raw: str, model: str, mode: str) -> tuple[str, list[str]]:
    print(f"\n[meta] extracting curriculum metadata...", end=" ", flush=True)
    start = time.time()

    response: ChatResponse = chat(
        model=model,
        options={"temperature": 0, "num_ctx": 8192, "num_predict": 4096},
        messages=[
            {"role": "system", "content": META_SYSTEM},
            {"role": "user",   "content": META_USER.format(raw=raw)},
        ],
    )

    raw_json = _sanitize_json(response.message.content)
    print(f"done ({time.time() - start:.2f}s)")

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        print(f"[error] frontmatter JSON parse failed: {e}")
        print(f"[debug] raw response:\n{raw_json}")
        exit(1)

    topics = data.get("topics", [])

    # inject fixed fields — never left to model
    data["generated_on"] = datetime.now().isoformat(timespec="seconds")
    data["model"]        = model
    data["mode"]         = mode

    return _dict_to_yaml(data), topics


def _dict_to_yaml(data: dict) -> str:
    lines = ["---"]
    for key, val in data.items():
        if isinstance(val, list):
            lines.append(f"{key}:")
            for item in val:
                lines.append(f'  - "{item}"')
        elif isinstance(val, bool):
            lines.append(f"{key}: {str(val).lower()}")
        elif isinstance(val, (int, float)):
            lines.append(f"{key}: {val}")
        else:
            lines.append(f'{key}: "{str(val).replace(chr(34), chr(92)+chr(34))}"')
    lines.append("---")
    return "\n".join(lines) + "\n"


# ── call 2: study plan (streaming) ───────────────────────────────────────────

def generate_plan(raw: str, lang: str, model: str) -> str:
    lang_name = LANG_INSTRUCTION[lang]
    print(f"\n[plan] model={model} lang={lang_name}")
    print(f"[plan] streaming...\n")
    print("-" * 56)

    start = time.time()
    chunks = []
    stream = chat(
        model=model,
        options={"temperature": 0.4, "num_ctx": 32768, "num_predict": 65536},
        messages=[
            {"role": "system", "content": PLAN_SYSTEM},
            {"role": "user",   "content": PLAN_USER.format(lang_name=lang_name, raw=raw)},
        ],
        stream=True,
    )

    for chunk in stream:
        token = chunk.message.content
        print(token, end="", flush=True)
        chunks.append(token)

    body = "".join(chunks)
    print(f"\n" + "-" * 56)
    print(f"[plan] done ({time.time() - start:.2f}s, {len(body)} chars)")
    return body


# ── call 2b: topic extraction from plan (json, no stream) ────────────────────

def extract_topics_from_plan(plan_body: str, model: str) -> list[str]:
    print(f"\n[topics] extracting expanded topic list...", end=" ", flush=True)
    start = time.time()

    response: ChatResponse = chat(
        model=model,
        options={"temperature": 0, "num_ctx": 32768, "num_predict": 2048},
        messages=[
            {"role": "system", "content": TOPIC_EXTRACT_SYSTEM},
            {"role": "user",   "content": TOPIC_EXTRACT_USER.format(plan=plan_body)},
        ],
    )

    raw_json = _sanitize_json(response.message.content)

    try:
        topics = json.loads(raw_json)
        if not isinstance(topics, list):
            raise ValueError("expected a JSON array")
        topics = [str(t) for t in topics if t]
        print(f"done ({time.time() - start:.2f}s)")
        print(f"[topics] {len(topics)} topics extracted")
        return topics
    except Exception as e:
        print(f"warn: {e}")
        print(f"[topics] extraction failed — falling back to raw curriculum topics")
        return []


# ── call 3: study material (streaming) ───────────────────────────────────────

def generate_material(raw: str, topics: list[str], lang: str, model: str) -> str:
    lang_name  = LANG_INSTRUCTION[lang]
    topics_str = "\n".join(f"{i+1}. {t}" for i, t in enumerate(topics))

    print(f"\n[material] model={model} lang={lang_name} topics={len(topics)}")
    print(f"[material] streaming...\n")
    print("-" * 56)

    start = time.time()
    chunks = []
    stream = chat(
        model=model,
        options={"temperature": 0.3, "num_ctx": 65536, "num_predict": 131072},
        messages=[
            {"role": "system", "content": MATERIAL_SYSTEM},
            {"role": "user",   "content": MATERIAL_USER.format(
                lang_name=lang_name,
                topics=topics_str,
                raw=raw,
            )},
        ],
        stream=True,
    )

    for chunk in stream:
        token = chunk.message.content
        print(token, end="", flush=True)
        chunks.append(token)

    body = "".join(chunks)
    print(f"\n" + "-" * 56)
    print(f"[material] done ({time.time() - start:.2f}s, {len(body)} chars)")
    return body


# ── cache ─────────────────────────────────────────────────────────────────────

def find_cached(input_file: str, model: str, lang: str, mode: str) -> Path | None:
    output_dir = Path("./outputs")
    if not output_dir.exists():
        return None
    basename = os.path.basename(input_file)
    pattern  = "*-study_plan.md" if mode == "plan" else "*-full.md"
    for path in sorted(output_dir.glob(pattern), reverse=True):
        meta = _read_frontmatter(path)
        if (meta.get("source") == basename
                and meta.get("model") == model
                and meta.get("lang") == lang
                and meta.get("mode") == mode):
            print(f"[cache] found existing output: {path}")
            return path
    return None


def _read_frontmatter(path: Path) -> dict:
    meta = {}
    try:
        content = path.read_text(encoding="utf-8")
        parts = content.split("---", 2)
        if len(parts) < 3:
            return meta
        for line in parts[1].strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip().strip('"')
    except Exception:
        pass
    return meta


# ── output ────────────────────────────────────────────────────────────────────

def save_output(frontmatter: str, sections: list[tuple[str, str]],
                input_file: str, model: str, lang: str, mode: str) -> Path:
    os.makedirs("./outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    slug      = re.sub(r"[^\w]+", "_", Path(input_file).stem.lower()).strip("_")[:40]
    suffix    = "study_plan" if mode == "plan" else "full"
    path      = Path(f"./outputs/{timestamp}-{slug}-{suffix}.md")

    fm = frontmatter.rstrip().removesuffix("---")
    fm += f'\nsource: "{os.path.basename(input_file)}"\n'
    fm += f'lang: "{lang}"\n'
    fm += "---\n"

    body = "\n\n---\n\n".join(f"# {label}\n\n{content}" for label, content in sections)
    path.write_text(fm + "\n" + body, encoding="utf-8")
    print(f"[output] saved → {path}")
    return path


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Curriculum → Self-Study Plan / Full Material")
    parser.add_argument("file",    nargs="?", help="Path to curriculum .txt file")
    parser.add_argument("--model", type=str, help="Skip model selection")
    parser.add_argument("--lang",  type=str, default=None, help="Output language code")
    parser.add_argument("--mode",  type=str, choices=MODES, default=None,
                        help="plan = study plan only | full = plan + material (default: ask)")
    return parser.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    input_path = Path(args.file if args.file else ask_file())
    if not input_path.exists():
        print(f"[error] file not found: {input_path}")
        exit(1)

    raw = input_path.read_text(encoding="utf-8").strip()

    # model selection
    if args.model:
        model = args.model
    else:
        models = list_models()
        if not models:
            print("[error] no matching models found. check MODEL_KEYWORDS.")
            exit(1)
        model = ask_model(models)

    # language selection
    lang = args.lang if args.lang in LANG_INSTRUCTION else None
    if lang is None:
        lang = ask_language()

    # mode selection
    mode = args.mode if args.mode in MODES else ask_mode()

    # cache check
    if find_cached(str(input_path), model, lang, mode):
        print("[done]")
        exit(0)

    # call 1 — frontmatter (fast, no stream)
    frontmatter, topics = generate_frontmatter(raw, model, mode)
    print(frontmatter)

    # call 2 — study plan (streaming)
    plan_body = generate_plan(raw, lang, model)

    sections = [("Study Plan", plan_body)]

    # call 2b — expand topic list from plan (fast, no stream)
    # call 3   — study material (streaming, only if full mode)
    if mode == "full":
        expanded = extract_topics_from_plan(plan_body, model)
        material_topics = expanded if expanded else topics
        material_body = generate_material(raw, material_topics, lang, model)
        sections.append(("Study Material", material_body))

    eject_model(model)
    save_output(frontmatter, sections, str(input_path), model, lang, mode)
    print("\n[done]")