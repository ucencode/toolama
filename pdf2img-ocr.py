#!/usr/bin/env python3

import os
import subprocess
import time
import argparse
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
- for images or diagrams, describe with [image: clear description of the visual content]
- for screenshots of application interfaces or terminal output, transcribe all visible text within the screenshot as-is, prefixed with [screenshot: brief description of the application/interface]
- ignore decorative elements: background images, borders, watermarks, repeated logos, slide templates
- if the image contains no text, return [no text detected]
- output only the transcribed text and permitted markers ([unclear: ...], [illegible], [image: ...], [screenshot: ...], [no text detected])
- do not interpret, explain, or summarize beyond what is specified above"""

REFINE_PROMPTS = {
    "clean": """Clean the following OCR text:
- fix OCR artifacts: misread characters (l/1, O/0, rn/m), broken words, stray symbols
- fix obvious grammar/spelling errors caused by OCR, not the original author
- merge sentences split across page boundaries
- remove repeated headers, footers, and page numbers
- preserve the original structure: headings, lists, paragraphs
- do NOT rephrase, summarize, or add content
- do NOT change the author's word choices or style

Return clean readable text.""",

    "summary": """Convert this into concise study notes:
- organize by topic, not by page order
- group related ideas under a heading, then 5–8 bullets per heading
- keep only key ideas and practical examples from the source
- drop abstract filler and non-essential explanations
- avoid academic language — use plain, direct wording
- make it easy to scan and understand quickly""",

    "deep": """Transform the content into a comprehensive, book-style document. Structure it as follows:

# [Document Title]

## Introduction
Brief overview of what this document covers and why it matters.

## [Topic Section]
For each major topic or concept found in the content:

### [Subtopic / Key Concept]
- Write in full prose paragraphs, not bullet points
- Explain the concept thoroughly with context
- Include real-world examples and analogies to aid understanding
- Clarify the "why" behind each idea, not just the "what"
- Connect ideas to each other where relevant

## Summary
Recap the most important takeaways in a few paragraphs.

Guidelines:
- Use proper Markdown headings (##, ###) to reflect document hierarchy
- Write in clear, plain language — avoid academic jargon
- Preserve all key information from the source; do not omit details
- Expand on ideas where needed to make them fully understandable
- Do not use excessive bullet points — prefer flowing prose"""
}

LANG_INSTRUCTION = {
    "en": "Respond and deliver the output in English.",
    "id": "Respond and deliver the output in Bahasa Indonesia."
}

VISION_MODEL_KEYWORDS = ["qwen3.5", "qwen3-vl", "qwen2.5vl", "deepseek-ocr", "llama3.2-vision", "gemma4", "ministral-3", "glm-ocr"]
REFINE_MODEL_KEYWORDS  = ["glm-5.1", "gemma4", "qwen3.5", "gpt-oss"]

REFINE_TEMPERATURE = {"clean": 0, "summary": 0, "deep": 0.4}
REFINE_MAX_TOKENS = {"clean": 8192, "summary": 4096, "deep": 16384}


# ── ollama model discovery ─────────────────────────────────
def list_models(keywords: list[str]) -> list[str]:
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    models = []
    for line in result.stdout.strip().splitlines()[1:]:
        name = line.split()[0]
        if any(kw in name for kw in keywords):
            models.append(name)
    return models

# ── stage 1: OCR ───────────────────────────────────────────
def ocr_pdf(path: str, ocr_model: str, dpi: int = 200, num_predict: int = 4096) -> tuple[list[str], int, int]:
    print(f"[init] loading PDF: {path}")
    start_total = time.time()

    pages = convert_from_path(path, dpi=dpi)
    print(f"[init] {len(pages)} pages found, dpi={dpi}")
    print(f"[init] model: {ocr_model}\n")

    full_text = []
    total_tokens = 0

    for i, page in enumerate(pages):
        text, tokens = extract_page(i, page, len(pages), ocr_model, num_predict)
        total_tokens += tokens
        full_text.append(f"--- Page {i+1} ---\n{text}")

    print(f"\n[ocr] completed in {time.time() - start_total:.2f}s")
    return "\n\n".join(full_text), total_tokens, len(pages)


def extract_page(i: int, page, total_pages: int, ocr_model: str, num_predict: int = 4096) -> tuple[str, int]:
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
            options={"temperature": 0, "num_ctx": 8192, "num_predict": num_predict},
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
    print("\nLanguage for compiled output? (en / id) [default: id]")
    lang = input(">>> ").strip().lower() or "id"
    return lang if lang in ("en", "id") else "id"


# ── stage 2: refine ────────────────────────────────────────
def refine(text: str, mode: str, lang: str, model: str) -> str:
    prompt = REFINE_PROMPTS[mode] + "\n\n" + LANG_INSTRUCTION[lang]
    temp = REFINE_TEMPERATURE.get(mode, 0)
    max_tokens = REFINE_MAX_TOKENS.get(mode, 8192)
    print(f"\n[refine] mode={mode} lang={lang} model={model} temp={temp} max_tokens={max_tokens}")
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
    parser.add_argument("--num-predict", type=int, default=4096, help="Max tokens the model can generate per OCR page. Raise to 8192+ for dense textbook pages to avoid silent truncation (default: 4096)")
    return parser.parse_args()


# ── output ─────────────────────────────────────────────────
def save_raw(text: str, timestamp: str, file: str, pages: int, dpi: int, model: str):
    os.makedirs("./outputs", exist_ok=True)
    raw_path = f"./outputs/{timestamp}-raw.txt"
    metadata = f"""---
file: {os.path.basename(file)}
timestamp: {timestamp}
pages: {pages}
dpi: {dpi}
model: {model}
---

"""
    with open(raw_path, "w") as f:
        f.write(metadata + text)
    print(f"[output] raw      → {raw_path}")


def save_refined(text: str, timestamp: str):
    os.makedirs("./outputs", exist_ok=True)
    compiled_path = f"./outputs/{timestamp}-compiled.txt"
    with open(compiled_path, "w") as f:
        f.write(text)
    print(f"[output] compiled → {compiled_path}")


# ── main ───────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # pick vision model
    vision_models = list_models(VISION_MODEL_KEYWORDS)
    if not vision_models:
        print("[error] no vision models found. check VISION_MODEL_KEYWORDS.")
        exit(1)
    ocr_model = ask_model(vision_models, label="vision model")

    text, token, page_count = ocr_pdf(args.file, ocr_model=ocr_model, dpi=args.dpi, num_predict=args.num_predict)
    eject_model(ocr_model)
    save_raw(text, timestamp, file=args.file, pages=page_count, dpi=args.dpi, model=ocr_model)

    mode = ask_mode(text.split("\n\n"), token)

    if mode != "skip":
        refine_models = list_models(REFINE_MODEL_KEYWORDS)
        if not refine_models:
            print("[error] no refine models found. check REFINE_MODEL_KEYWORDS.")
        else:
            model = ask_model(refine_models, label="refine model")
            lang = ask_language()
            compiled_text = refine(text, mode, lang, model)
            save_refined(compiled_text, timestamp)

    print("\n[done]")