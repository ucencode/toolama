# slide-to-doc

Offline AI toolkit for converting PDFs and slides into structured documents and self-study material, powered by Ollama and local LLMs.

## Project Structure

```
slide-to-doc/
├── inputs/               # drop PDF and curriculum .txt files here
├── outputs/
│   ├── ocr/              # raw + compiled OCR outputs
│   └── generator/        # generated study plans and materials
├── tools/
│   ├── ocr/              # PDF → text pipeline
│   │   ├── pdf2img-ocr.py
│   │   ├── batch-pdf2img-ocr.py
│   │   └── presets/
│   └── generator/        # curriculum → study plan / material
│       └── generator.py
├── venv/
└── setup.sh
```

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running

## Setup

```bash
bash setup.sh
source venv/bin/activate
```

---

## Tool 1 — OCR (`tools/ocr/`)

Converts PDF pages to images, runs OCR via a vision model, and optionally refines the output into clean text, study notes, or structured book-style documents.

### Supported Models

| Role | Keywords matched |
|------|-----------------|
| Vision (OCR) | `qwen3.5`, `qwen3-vl`, `qwen2.5vl`, `deepseek-ocr`, `llama3.2-vision`, `gemma4`, `ministral-3`, `glm-ocr` |
| Refine (LLM) | `glm-5.1`, `gemma4`, `qwen3.5`, `gpt-oss` |

```bash
ollama pull glm-ocr:bf16
```

### Usage

**Interactive:**

```bash
python tools/ocr/pdf2img-ocr.py inputs/slides.pdf
```

Prompts you to select vision model, refine mode, language, and audience level. Use shell tab completion on the file path.

**With preset (non-interactive):**

```bash
python tools/ocr/pdf2img-ocr.py inputs/slides.pdf --preset example.toml
```

**Batch — process all PDFs in `inputs/`:**

```bash
python tools/ocr/batch-pdf2img-ocr.py --preset example.toml
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dpi` | `200` | Render resolution (higher = more detail, slower) |
| `--preset` | — | Load a TOML preset from `tools/ocr/presets/`, skip interactive prompts |

### Presets

TOML files in `tools/ocr/presets/`. See [`tools/ocr/presets/example.toml`](tools/ocr/presets/example.toml):

```toml
vision_model = "qwen3.5:9b"
refine_model = "gpt-oss:120b-cloud"
action = "deep"      # clean | summary | deep | skip
lang = "en"          # "auto" (preserve source language) or see supported languages below
level = "beginner"   # beginner | intermediate | advanced
```

Validation runs before processing — invalid models, actions, languages, or levels will exit with a clear error.

### Refine Modes

| Mode | Description |
|------|-------------|
| `skip` | Save raw OCR only |
| `clean` | Fix OCR noise, broken words, grammar |
| `summary` | Compress into bullet-point study notes |
| `deep` | Book-style structured document with prose and analogies |

### Output

```
outputs/ocr/
  <timestamp>-raw.txt       # raw OCR text (file, pages, dpi, model)
  <timestamp>-compiled.txt  # refined output (origin, model, mode, lang, level)
```

Raw OCR results are cached per PDF filename and vision model. Re-running the same file with the same model skips OCR and reuses the cached output.

---

## Tool 2 — Learning Plan Generator (`tools/generator/`)

Takes a curriculum `.txt` file and generates a structured self-study plan and per-topic study material using a local LLM.

### Supported Models

Any model matched by keywords: `llama3`, `qwen3`, `gemma`, `mistral`, `deepseek`, `phi`, `gpt-oss`

### Usage

**Interactive (picks curriculum file from `inputs/`):**

```bash
python tools/generator/generator.py
```

Prompts you to select a model, output language, and mode (plan only or plan + material).

**With arguments:**

```bash
python tools/generator/generator.py inputs/curriculum.txt --lang en --mode full
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | — | Skip model selection prompt |
| `--lang` | — | Output language code (see supported languages) |
| `--mode` | — | `plan` = study plan only \| `full` = plan + material per topic |

### Modes

| Mode | Output |
|------|--------|
| `plan` | Weekly schedule, phase breakdown, per-topic checkpoints, resource recommendations |
| `full` | Everything in `plan` + per-topic study material (concept, worked examples, practice problems, misconceptions, go deeper) |

### Output

```
outputs/generator/
  <timestamp>-<slug>-study_plan.md   # plan mode
  <timestamp>-<slug>-full.md         # full mode
```

All output files include YAML frontmatter (course, topics, credits, estimated weeks, model, language).

---

## Supported Languages (both tools)

| Code | Language |
|------|----------|
| `auto` | Preserve source language |
| `ar` | العربية (Arabic) |
| `de` | Deutsch (German) |
| `en` | English |
| `es` | Español (Spanish) |
| `fi` | Suomi (Finnish)* |
| `fr` | Français (French) |
| `hi` | हिन्दी (Hindi)* |
| `id` | Bahasa Indonesia |
| `it` | Italiano (Italian)* |
| `ja` | 日本語 (Japanese)* |
| `ko` | 한국어 (Korean)* |
| `nl` | Nederlands (Dutch)* |
| `pl` | Polski (Polish)* |
| `pt` | Português (Portuguese) |
| `ru` | Русский (Russian) |
| `sv` | Svenska (Swedish)* |
| `th` | ภาษาไทย (Thai)* |
| `tr` | Türkçe (Turkish)* |
| `uk` | Українська (Ukrainian)* |
| `vi` | Tiếng Việt (Vietnamese)* |
| `zh` | 简体中文 (Chinese) |

\* Output quality depends on the model's proficiency in this language.
