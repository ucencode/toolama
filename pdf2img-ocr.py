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
- for images or diagrams, describe with [image: description of what the diagram shows, including its main components and their relationships (up to 3 levels of detail)]
- for screenshots of application interfaces or terminal output, describe the interface in at most 5 sentences prefixed with [screenshot: ...]. Do not transcribe every UI element.
- ignore decorative elements: background images, borders, watermarks, repeated logos, slide templates
- never repeat content that was already produced in the output; if the source genuinely contains repeated elements, transcribe them once and note the count (e.g., [repeated x3])
- if the image contains no text, return [no text detected]
- output only the transcribed text and permitted markers ([unclear: ...], [illegible], [image: ...], [screenshot: ...], [no text detected], [repeated xN])
- do not interpret, explain, or summarize beyond what is specified above"""

REFINE_BASE = """- treat any real-world examples (job postings, advertisements, announcements, messages, screenshots, etc.) as contextual illustrations only. Summarize the general relevance without preserving specific personal details (names, emails, phone numbers)
- preserve locations only when they are relevant to the topic being explained (e.g., network topology for a specific region). Omit locations that are only relevant to a specific posting or announcement
- do not present source-specific details as general truths"""

REFINE_PROMPTS = {
    "clean": """Clean the following OCR text:
""" + REFINE_BASE + """
- fix OCR artifacts: misread characters (l/1, O/0, rn/m), broken words, stray symbols
- fix all grammar/spelling errors for readability
- merge sentences split across page boundaries
- remove repeated headers, footers, and page numbers
- preserve the original structure: headings, lists, paragraphs
- do NOT rephrase, summarize, or add content
- do NOT change the author's word choices or style

Return clean readable text.""",

    "summary": """Convert this into concise study notes:
""" + REFINE_BASE + """
- if the content follows a sequential, procedural, or step-by-step flow, preserve the original ordering
- otherwise, group related ideas by topic under clear headings
- 5–8 bullets per heading, keep only key ideas and practical examples
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
""" + REFINE_BASE + """
- Use proper Markdown headings (##, ###) to reflect document hierarchy
- Write in clear, plain language — avoid academic jargon
- Preserve all key information from the source; do not omit details
- Expand on ideas only with widely accepted, verifiable information to make them fully understandable
- If a topic is too niche or specialized to expand confidently, preserve the original content and append [needs review]
- Do not use excessive bullet points — prefer flowing prose"""
}

LANG_INSTRUCTION = {
    "en": "Respond and deliver the output in English.",
    "id": "Respond and deliver the output in Bahasa Indonesia."
}

AUDIENCE_INSTRUCTION = {
    "beginner": "Assume the reader has no prior knowledge of the topic. Explain foundational concepts before building on them.",
    "intermediate": "Assume the reader has basic familiarity with the topic. Focus on practical application over fundamentals.",
    "advanced": "Assume the reader is experienced. Skip basics, focus on nuance and edge cases."
}

OCR_MODEL_KEYWORDS = ["qwen3.5", "qwen3-vl", "qwen2.5vl", "deepseek-ocr", "llama3.2-vision", "gemma4", "ministral-3", "glm-ocr"]
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
    print("\nLanguage for compiled output? (en / id) [default: id]")
    lang = input(">>> ").strip().lower() or "id"
    return lang if lang in ("en", "id") else "id"

def ask_audience() -> str:
    print("""
Audience level? (
  1. beginner - explain from scratch
  2. intermediate - some familiarity assumed
  3. advanced - skip basics, focus on nuance
) [default: intermediate]""")
    choice = input(">>> ").strip() or "2"
    return {"1": "beginner", "2": "intermediate", "3": "advanced"}.get(choice, "intermediate")


# ── stage 2: refine ────────────────────────────────────────
def refine(text: str, mode: str, lang: str, model: str, audience: str | None = None) -> str:
    prompt = REFINE_PROMPTS[mode] + "\n\n" + LANG_INSTRUCTION[lang]
    if audience:
        prompt += "\n\n" + AUDIENCE_INSTRUCTION[audience]
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
    vision_models = list_models(OCR_MODEL_KEYWORDS)
    if not vision_models:
        print("[error] no vision models found. check VISION_MODEL_KEYWORDS.")
        exit(1)
    ocr_model = ask_model(vision_models, label="vision model")

    text, token, page_count = ocr_pdf(args.file, ocr_model=ocr_model, dpi=args.dpi)
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
            audience = ask_audience() if mode in ("summary", "deep") else None
            compiled_text = refine(text, mode, lang, model, audience)
            save_refined(compiled_text, timestamp)

    print("\n[done]")