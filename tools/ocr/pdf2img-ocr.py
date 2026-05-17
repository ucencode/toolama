#!/usr/bin/env python3

import os
import subprocess
import time
import argparse
import tomllib
from pathlib import Path
from datetime import datetime
from io import BytesIO

from pdf2image import convert_from_path
from ollama import chat, ChatResponse

# OCR_PROMPT = """You are an expert OCR system. Transcribe all text from this image accurately.
# - preserve original structure, hierarchy, and layout
# - include all visible text: titles, subtitles, bullets, labels, captions
# - maintain list formatting and indentation where present
# - reproduce tables using plain text or markdown table format
# - for multi-column layouts, transcribe left to right, top to bottom
# - do not interpret, explain, or summarize
# - do not add commentary or descriptions of images/diagrams
# - if text is partially visible or unclear, make best effort
# - output only the transcribed text, nothing else"""

OCR_PROMPT = """You are an expert OCR system. Transcribe all text from this image accurately.
- preserve original structure, hierarchy, and layout
- include all visible text: titles, subtitles, bullets, labels, captions
- maintain list formatting and indentation where present
- reproduce tables using markdown table format
- for multi-column layouts, transcribe left to right, top to bottom
- if text is partially visible or unclear, mark with [unclear: best guess] or [illegible]
- for images or diagrams, describe with [image: description of what the diagram shows, including its main components and their relationships (up to 3 levels of detail)]
- for screenshots of application interfaces or terminal output, describe the interface in at most 5 sentences prefixed with [screenshot: ...]. Do not transcribe every UI element.
- if a diagram contains more than 20 distinct labeled elements, describe it as [image: ...] only — do not attempt to enumerate all elements
- ignore decorative elements: background images, borders, watermarks, repeated logos, slide templates
- never repeat content that was already produced in the output; if the source genuinely contains repeated elements, transcribe them once and note the count (e.g., [repeated x3])
- if the image contains no text, return [no text detected]
- output only the transcribed text and permitted markers ([unclear: ...], [illegible], [image: ...], [screenshot: ...], [no text detected], [repeated xN])
- do not interpret, explain, or summarize beyond what is specified above"""

REFINE_BASE = """- you are processing OCR output from presentation slides (lectures, meetings, competitions, pitches, or similar)
- do NOT present source-specific details as general truths
- treat real-world examples (job postings, ads, announcements, screenshots, etc.) as contextual illustrations only — summarize relevance without preserving personal details (names, emails, phone numbers)
- preserve locations only when relevant to the topic being explained; omit if tied only to a specific posting or announcement"""

REFINE_PROMPTS = {
    "clean": """Clean the following OCR text from presentation slides.

""" + REFINE_BASE + """

Cleaning rules:
- fix OCR artifacts: misread characters (l/1, O/0, rn/m), broken words, stray symbols
- fix grammar and spelling errors only where meaning is unclear or readability is significantly affected
- preserve page boundary markers (--- Page N ---) and all OCR markers ([image: ...], [unclear: ...], [repeated xN]) exactly as-is
- preserve original structure: headings, lists, paragraphs, indentation
- remove repeated headers, footers, and page numbers only if they are clearly decorative or auto-generated
- do NOT merge content across page boundaries
- do NOT rephrase, summarize, or add content
- do NOT change the author's word choices or style

Return clean, readable text with structure intact.""",

    "summary": """Convert the following presentation slide content into concise study notes.

""" + REFINE_BASE + """

Summary rules:
- omit [image: ...] markers — extract meaning only if the diagram description contains relevant information
- if content follows a sequential or procedural flow, preserve that ordering
- otherwise, group related ideas by topic under clear headings
- 5–8 bullets per heading; keep only key ideas and practical examples
- drop abstract filler and non-essential explanations
- use plain, direct wording — avoid academic or formal language
- make it easy to scan and review quickly""",

    "deep": """Transform the following presentation slide content into a comprehensive, book-style document.

""" + REFINE_BASE + """

Output structure:
# [Document Title]

## Introduction
Brief overview of what this document covers and why it matters.

## [Topic Section]
For each major topic or concept found in the content:

### [Subtopic / Key Concept]
Write in full prose paragraphs. Explain the concept thoroughly with context. Include
real-world examples and analogies. Clarify the "why" behind each idea, not just the
"what". Connect ideas to each other where relevant.

## Summary
Recap the most important takeaways in a few paragraphs.

Writing rules:
- treat [image: ...] descriptions as source content — expand on what the diagram illustrates
- use proper Markdown headings (##, ###) to reflect document hierarchy
- write in clear, plain language — avoid academic jargon
- preserve all key information from the source; do not omit details
- expand on ideas only with widely accepted, verifiable information
- if a topic is too niche to expand confidently, preserve original content and append [needs review]
- prefer flowing prose over bullet points"""
}

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

LANG_EXPERIMENTAL = {"ja", "ko", "it", "nl", "pl", "tr", "hi", "vi", "uk", "fi", "sv", "th"}

AUDIENCE_INSTRUCTION = {
    "beginner": "Assume the reader has no prior knowledge of the topic. Explain foundational concepts before building on them.",
    "intermediate": "Assume the reader has basic familiarity with the topic. Focus on practical application over fundamentals.",
    "advanced": "Assume the reader is experienced. Skip basics, focus on nuance and edge cases."
}

OCR_MODEL_KEYWORDS = ["qwen3.5", "qwen3-vl", "qwen2.5vl", "deepseek-ocr", "llama3.2-vision", "gemma4", "ministral-3", "glm-ocr"]
REFINE_MODEL_KEYWORDS  = ["glm-5.1", "gemma4", "qwen3.5", "gpt-oss"]

OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "ocr"

REFINE_TEMPERATURE = {"clean": 0, "summary": 0, "deep": 0.4}

REFINE_MAX_TOKENS = {"clean": 65536, "summary": 65536, "deep": 131072}


# ── ollama model discovery ─────────────────────────────────
def list_models(keywords: list[str]) -> list[str]:
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    models = []
    for line in result.stdout.strip().splitlines()[1:]:
        name = line.split()[0]
        if any(kw in name for kw in keywords):
            models.append(name)
    return models


def _get_all_ollama_models() -> set[str]:
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[error] 'ollama list' failed: {result.stderr.strip()}")
        exit(1)
    models = set()
    for line in result.stdout.strip().splitlines()[1:]:
        name = line.split()[0]
        models.add(name)
    return models


# ── stage 1: OCR ───────────────────────────────────────────
def ocr_pdf(path: str, ocr_model: str, dpi: int = 200) -> tuple[list[str], int, int]:
    print(f"[init] loading PDF: {path}")
    start_total = time.time()

    pages = convert_from_path(path, dpi=dpi)
    print(f"[init] {len(pages)} pages found, dpi={dpi}")
    print(f"[init] model: {ocr_model}\n")

    full_text = []
    total_tokens = 0

    for i, page in enumerate(pages):
        text, tokens = extract_page(i, page, len(pages), ocr_model)
        total_tokens += tokens
        full_text.append(f"--- Page {i+1} ---\n{text}")

    print(f"\n[ocr] completed in {time.time() - start_total:.2f}s")
    return "\n\n".join(full_text), total_tokens, len(pages)


def extract_page(i: int, page, total_pages: int, ocr_model: str) -> tuple[str, int]:
    page_start = time.time()
    print(f"[page {i+1}/{total_pages}] encoding image...", end=" ", flush=True)

    buf = BytesIO()
    page.save(buf, format="PNG")
    raw = buf.getvalue()
    print(f"done ({len(raw)/1024:.1f} KB)")

    print(f"[page {i+1}/{total_pages}] sending to {ocr_model}...", end=" ", flush=True)
    try:
        response: ChatResponse = chat(
            model=ocr_model,
            options={"temperature": 0, "num_ctx": 8192, "num_predict": get_ocr_max_tokens(len(raw) / 1024)},
            stream=False,
            think=False,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert OCR system specialized in extracting text from slides and documents."
                },
                {
                    "role": "user",
                    "content": OCR_PROMPT,
                    "images": [raw]
                }
            ]
        )
    except Exception as e:
        print(f"\n[page {i+1}] error: {e}")
        return f"[missing page {i+1}]", 3  # return placeholder text and token count for error case

    text = response.message.content
    tokens = response.eval_count or 0
    elapsed = time.time() - page_start
    print(f"done ({elapsed:.2f}s, {len(text)} chars, {tokens} tokens)")
    return text, tokens

def get_ocr_max_tokens(image_size_kb: int) -> int:
    if image_size_kb < 500:
        return 1024
    elif image_size_kb < 1000:
        return 2048
    else:
        return 4096

# ── stage 1.5: eject ──────────────────────────────────────
def eject_model(model: str):
    print(f"[ollama] stopping {model}...", end=" ", flush=True)
    result = subprocess.run(["ollama", "stop", model], capture_output=True, text=True)
    print("done" if result.returncode == 0 else f"warning: {result.stderr.strip()}")


# ── interactive prompts ────────────────────────────────────
def ask_mode(raw_pages: list[str], total_tokens: int) -> str:
    total_chars = sum(len(p) for p in raw_pages)
    approx_tokens = total_tokens 

    print(f"\nOCR result: {total_chars:,} chars (~{approx_tokens:,} tokens)")
    print("""
Refine output? (
  1. skip - do nothing
  2. clean - fix OCR mess only
  3. summary - compress into notes
  4. deep - structured + analogy + understanding
) [default: skip]""")
    choice = input(">>> ").strip() or "1"
    return {"1": "skip", "2": "clean", "3": "summary", "4": "deep"}.get(choice, "skip")

# rename ask_model to be generic
def ask_model(models: list[str], label: str = "model") -> str:
    print(f"\nSelect {label}:")
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
    print(f"\nLanguage for compiled output? ({' / '.join(codes)}) [default: auto]")
    lang = input(">>> ").strip().lower() or "auto"
    if lang not in LANG_INSTRUCTION:
        print(f"[warn] unknown language '{lang}', using auto")
        return "auto"
    if lang in LANG_EXPERIMENTAL:
        print(f"[warn] '{LANG_INSTRUCTION[lang]}' output quality depends on the refine model's proficiency in this language. Results may vary.")
    return lang

def ask_audience() -> str:
    print("""
Audience level? (
  1. beginner - explain from scratch
  2. intermediate - some familiarity assumed
  3. advanced - skip basics, focus on nuance
) [default: 2]""")
    choice = input(">>> ").strip() or "2"
    return {"1": "beginner", "2": "intermediate", "3": "advanced"}.get(choice, "intermediate")


# ── stage 2: refine ────────────────────────────────────────
def refine(text: str, mode: str, lang: str, model: str, audience: str | None = None) -> str:
    lang_name = LANG_INSTRUCTION[lang]
    prompt = REFINE_PROMPTS[mode] + "\n\n" + f"Respond and deliver the output in {lang_name}."
    if audience:
        prompt += "\n\n" + AUDIENCE_INSTRUCTION[audience]
    temp = REFINE_TEMPERATURE.get(mode, 0)
    max_tokens = REFINE_MAX_TOKENS.get(mode, 8192)
    level_str = f" audience={audience}" if audience else ""
    print(f"\n[refine] mode={mode} lang={lang_name}{level_str} model={model} temp={temp} max_tokens={max_tokens}")
    print(f"[refine] sending {len(text)} chars...", end=" ", flush=True)

    start = time.time()
    response: ChatResponse = chat(
        model=model,
        options={"temperature": temp, "num_predict": max_tokens, "num_ctx": 32768},
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ]
    )
    result = response.message.content
    print(f"done ({time.time() - start:.2f}s, {len(result)} chars)")
    return result


# ── args ───────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="PDF OCR Pipeline")
    parser.add_argument("file", help="Path to PDF file")
    parser.add_argument("--dpi", type=int, default=200, help="Render DPI (default: 200)")
    parser.add_argument("--preset", type=str, metavar="FILE",
                        help="Load preset from presets/<FILE>, skip interactive prompts")
    return parser.parse_args()


def load_preset(filename: str) -> dict:
    preset_dir = Path(__file__).parent / "presets"
    config_path = preset_dir / filename
    if not config_path.exists():
        available = [f.name for f in preset_dir.glob("*.toml")] if preset_dir.exists() else []
        print(f"[error] preset not found: {config_path}")
        if available:
            print(f"[hint] available presets: {', '.join(sorted(available))}")
        exit(1)
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def check_preset(config: dict, filename: str) -> None:
    required = {"vision_model", "action", "lang", "level"}
    missing = required - config.keys()
    if missing:
        print(f"[error] missing keys in {filename}: {', '.join(sorted(missing))}")
        exit(1)

    valid_actions = {"skip", "clean", "summary", "deep"}
    if config["action"] not in valid_actions:
        print(f"[error] invalid action '{config['action']}' in {filename}. Must be one of: {', '.join(sorted(valid_actions))}")
        exit(1)

    if config["lang"] not in LANG_INSTRUCTION:
        print(f"[error] invalid lang '{config['lang']}' in {filename}. Must be one of: {', '.join(LANG_INSTRUCTION.keys())}")
        exit(1)

    if config["level"] not in AUDIENCE_INSTRUCTION:
        print(f"[error] invalid level '{config['level']}' in {filename}. Must be one of: {', '.join(AUDIENCE_INSTRUCTION.keys())}")
        exit(1)

    available = _get_all_ollama_models()

    if config["vision_model"] not in available:
        print(f"[error] vision_model '{config['vision_model']}' not found in ollama. Available: {', '.join(sorted(available))}")
        exit(1)

    if config["action"] != "skip":
        if "refine_model" not in config:
            print(f"[error] missing key 'refine_model' in {filename} (required when action != skip)")
            exit(1)
        if config["refine_model"] not in available:
            print(f"[error] refine_model '{config['refine_model']}' not found in ollama. Available: {', '.join(sorted(available))}")
            exit(1)


# ── cache lookup ──────────────────────────────────────────
def find_existing_raw(pdf_file: str, ocr_model: str) -> str | None:
    """Search outputs/ for an existing raw OCR file matching the same PDF basename and model."""
    output_dir = OUTPUT_DIR
    if not output_dir.exists():
        return None
    basename = os.path.basename(pdf_file)
    for raw_path in sorted(output_dir.glob("*-raw.txt"), reverse=True):
        with open(raw_path) as f:
            meta = {}
            in_frontmatter = False
            for line in f:
                line = line.strip()
                if line == "---":
                    if in_frontmatter:
                        break
                    in_frontmatter = True
                    continue
                if not in_frontmatter:
                    break
                if ":" in line:
                    key, val = line.split(":", 1)
                    meta[key.strip()] = val.strip()
            if meta.get("file") == basename and meta.get("model") == ocr_model:
                print(f"[cache] found existing OCR: {raw_path}")
                return raw_path
    return None


def load_raw_text(raw_path) -> str:
    """Load text content from a raw OCR file, skipping the frontmatter."""
    with open(raw_path) as f:
        content = f.read()
    # skip frontmatter (between --- markers)
    parts = content.split("---", 2)
    if len(parts) >= 3:
        return parts[2].strip()
    return content


# ── output ─────────────────────────────────────────────────
def save_raw(text: str, timestamp: str, file: str, pages: int, dpi: int, model: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    raw_path = OUTPUT_DIR / f"{timestamp}-raw.txt"
    metadata = f"""---
file: {os.path.basename(file)}
timestamp: {timestamp}
pages: {pages}
dpi: {dpi}
model: {model}
---

"""
    with open(str(raw_path), "w") as f:
        f.write(metadata + text)
    print(f"[output] raw      → {raw_path}")


def save_refined(text: str, timestamp: str, *, origin: str, raw_file: str,
                  model: str, mode: str, lang: str, level: str | None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    compiled_path = OUTPUT_DIR / f"{timestamp}-compiled.txt"
    level_num = {"beginner": 1, "intermediate": 2, "advanced": 3}.get(level, "n/a")
    metadata = f"""---
origin: {origin}
file: {raw_file}
timestamp: {timestamp}
model: {model}
mode: {mode}
lang: {lang}
level: {level_num}
---

"""
    with open(str(compiled_path), "w") as f:
        f.write(metadata + text)
    print(f"[output] compiled → {compiled_path}")


# ── main ───────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    if args.preset:
        config = load_preset(args.preset)
        check_preset(config, args.preset)

        ocr_model = config["vision_model"]
        print(f"[config] vision_model={ocr_model} action={config['action']} lang={config['lang']} level={config['level']}")
        if config["lang"] in LANG_EXPERIMENTAL:
            print(f"[warn] '{LANG_INSTRUCTION[config['lang']]}' output quality depends on the refine model's proficiency in this language. Results may vary.")

        existing_raw = find_existing_raw(args.file, ocr_model)
        if existing_raw:
            text = load_raw_text(existing_raw)
            print(f"[cache] skipping OCR, loaded {len(text)} chars from {existing_raw}")
        else:
            text, token, page_count = ocr_pdf(args.file, ocr_model=ocr_model, dpi=args.dpi)
            eject_model(ocr_model)
            save_raw(text, timestamp, file=args.file, pages=page_count, dpi=args.dpi, model=ocr_model)

        mode = config["action"]
        if mode != "skip":
            refine_model = config["refine_model"]
            lang = config["lang"]
            audience = config["level"] if mode in ("summary", "deep") else None
            compiled_text = refine(text, mode, lang, refine_model, audience)
            raw_file = os.path.basename(str(existing_raw)) if existing_raw else f"{timestamp}-raw.txt"
            save_refined(compiled_text, timestamp, origin=os.path.basename(args.file),
                         raw_file=raw_file, model=refine_model, mode=mode, lang=lang, level=audience)
            eject_model(refine_model)
    else:
        # interactive flow
        vision_models = list_models(OCR_MODEL_KEYWORDS)
        if not vision_models:
            print("[error] no vision models found. check VISION_MODEL_KEYWORDS.")
            exit(1)
        ocr_model = ask_model(vision_models, label="vision model")

        existing_raw = find_existing_raw(args.file, ocr_model)
        if existing_raw:
            text = load_raw_text(existing_raw)
            print(f"[cache] skipping OCR, loaded {len(text)} chars from {existing_raw}")
            token = len(text) // 4  # rough estimate for display
        else:
            text, token, page_count = ocr_pdf(args.file, ocr_model=ocr_model, dpi=args.dpi)
            eject_model(ocr_model)
            save_raw(text, timestamp, file=args.file, pages=page_count, dpi=args.dpi, model=ocr_model)

        mode = ask_mode(text.split("\n\n"), token)

        if mode != "skip":
            refine_models = list_models(REFINE_MODEL_KEYWORDS)
            if not refine_models:
                print("[error] no refine models found. check REFINE_MODEL_KEYWORDS.")
            else:
                refine_model = ask_model(refine_models, label="refine model")
                lang = ask_language()
                audience = ask_audience() if mode in ("summary", "deep") else None
                compiled_text = refine(text, mode, lang, refine_model, audience)
                raw_file = os.path.basename(str(existing_raw)) if existing_raw else f"{timestamp}-raw.txt"
                save_refined(compiled_text, timestamp, origin=os.path.basename(args.file),
                             raw_file=raw_file, model=refine_model, mode=mode, lang=lang, level=audience)
                eject_model(refine_model)

    print("\n[done]")