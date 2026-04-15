# slide-to-doc

Offline PDF-to-document pipeline using Ollama and local LLMs. Converts PDF pages to images, runs OCR via a vision model, and optionally refines the output into clean text, study notes, or structured book-style documents.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running

## Setup

```bash
bash setup.sh
source venv/bin/activate
```

## Supported Models

| Role | Keywords matched |
|------|-----------------|
| Vision (OCR) | `qwen3.5`, `qwen3-vl`, `qwen2.5vl`, `deepseek-ocr`, `llama3.2-vision`, `gemma4`, `ministral-3`, `glm-ocr` |
| Refine (LLM) | `glm-5.1`, `gemma4`, `qwen3.5`, `gpt-oss` |

Pull a model example:

```bash
ollama pull glm-ocr:bf16
```

## Usage

### Interactive (default)

```bash
python pdf2img-ocr.py path/to/file.pdf
```

Prompts you to select vision model, refine mode, language, and audience level.

### With preset

```bash
python pdf2img-ocr.py path/to/file.pdf --preset example.toml
```

Loads config from `presets/<filename>` and skips all interactive prompts.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dpi` | `200` | Render resolution (higher = more detail, slower) |
| `--preset` | — | Load a TOML preset from `presets/`, skip interactive prompts |

## Presets

TOML files in the `presets/` directory. See [`presets/example.toml`](presets/example.toml):

```toml
vision_model = "qwen3.5:9b"
refine_model = "gpt-oss:120b-cloud"
action = "deep"      # clean | summary | deep | skip
lang = "en"          # en | id
level = "beginner"   # beginner | intermediate | advanced
```

Validation runs before processing — invalid models (not in `ollama list`), actions, languages, or levels will exit with a clear error.

## Refine Modes

| Mode | Description |
|------|-------------|
| `skip` | Save raw OCR only |
| `clean` | Fix OCR noise, broken words, grammar |
| `summary` | Compress into bullet-point study notes |
| `deep` | Book-style structured document with prose and analogies |

## Output

```
outputs/
  <timestamp>-raw.txt       <- raw OCR text per page
  <timestamp>-compiled.txt  <- refined output (if selected)
```
