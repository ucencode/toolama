#!/usr/bin/env python3

import os
import subprocess
import time
import argparse
from datetime import datetime
from io import BytesIO

from pdf2image import convert_from_path
from ollama import chat, ChatResponse

OCR_PROMPT = """You are an expert OCR system. Transcribe all text from this image accurately.
- preserve original structure, hierarchy, and layout
- include all visible text: titles, subtitles, bullets, labels, captions
- maintain list formatting and indentation where present
- do not interpret, explain, or summarize
- do not add commentary or descriptions of images/diagrams
- if text is partially visible or unclear, make best effort
- output only the transcribed text, nothing else"""

REFINE_PROMPTS = {
    "clean": """Clean the following OCR text:
- fix broken words and grammar
- remove noise and repetition
- keep original meaning
- do NOT summarize

Return clean readable text.""",

    "summary": """Convert this into concise study notes:
- use bullet points
- keep only key ideas
- remove explanations that are not essential
- max 5–8 bullets per concept
- avoid academic language
- use practical examples
- make it easy to understand quickly""",

    "deep": """Extract and structure the content into:

- Concept:
- Core Idea (1 sentence):
- Key Points:
- Real-world analogy:

Keep it concise but meaningful.
- avoid academic language
- use practical examples
- make it easy to understand quickly"""
}

LANG_INSTRUCTION = {
    "en": "Respond and deliver the output in English.",
    "id": "Respond and deliver the output in Bahasa Indonesia."
}

VISION_MODEL_KEYWORDS = ["qwen3-vl", "qwen2.5vl", "deepseek-ocr", "llama3.2-vision", "gemma4", "ministral-3", "glm-ocr"]
REFINE_MODEL_KEYWORDS  = ["glm-5.1", "gemma4", "qwen3.5", "gpt-oss"]


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
def ocr_pdf(path: str, dpi: int = 200) -> str:
    print(f"[init] loading PDF: {path}")
    start_total = time.time()

    pages = convert_from_path(path, dpi=dpi)
    print(f"[init] {len(pages)} pages found, dpi={dpi}")
    print(f"[init] model: {ocr_model}\n")

    full_text = []
    total_tokens = 0

    for i, page in enumerate(pages):
        page_start = time.time()
        print(f"[page {i+1}/{len(pages)}] encoding image...", end=" ", flush=True)

        buf = BytesIO()
        page.save(buf, format="PNG")
        raw = buf.getvalue()
        print(f"done ({len(raw)/1024:.1f} KB)")

        print(f"[page {i+1}/{len(pages)}] sending to {ocr_model}...", end=" ", flush=True)
        response: ChatResponse = chat(
            model=ocr_model,
            options={"temperature": 0, "think": False, "num_predict": 4096},
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

        text = response.message.content
        total_tokens += response.eval_count or 0  # output tokens from ollama
        
        elapsed = time.time() - page_start
        print(f"done ({elapsed:.2f}s, {len(text)} chars, {response.eval_count or 0} tokens)")
        full_text.append(f"--- Page {i+1} ---\n{text}")

    print(f"\n[ocr] completed in {time.time() - start_total:.2f}s")
    return {"text": "\n\n".join(full_text), "tokens": total_tokens}


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
    print(f"\n[refine] mode={mode} lang={lang} model={model}")
    print(f"[refine] sending {len(text)} chars...", end=" ", flush=True)

    start = time.time()
    response: ChatResponse = chat(
        model=model,
        options={"temperature": 0},
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
def save_outputs(raw: str, compiled: str | None, timestamp: str):
    os.makedirs("./outputs", exist_ok=True)

    raw_path = f"./outputs/{timestamp}-raw.txt"
    with open(raw_path, "w") as f:
        f.write(raw)
    print(f"[output] raw      → {raw_path}")

    if compiled:
        compiled_path = f"./outputs/{timestamp}-compiled.txt"
        with open(compiled_path, "w") as f:
            f.write(compiled)
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

    ocr_result = ocr_pdf(args.file, dpi=args.dpi)
    eject_model(ocr_model)

    mode = ask_mode(ocr_result["text"].split("\n\n"), ocr_result["tokens"])

    if mode == "skip":
        save_outputs(ocr_result["text"], None, timestamp)
    else:
        refine_models = list_models(REFINE_MODEL_KEYWORDS)
        if not refine_models:
            print("[error] no refine models found. check REFINE_MODEL_KEYWORDS.")
            save_outputs(ocr_result["text"], None, timestamp)
        else:
            model = ask_model(refine_models, label="refine model")
            lang = ask_language()
            compiled_text = refine(ocr_result["text"], mode, lang, model)
            save_outputs(ocr_result["text"], compiled_text, timestamp)

    print("\n[done]")