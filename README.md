# Spectrum News Agent

A Python agent that collects news articles from left, center, and right-leaning sources, extracts full text, summarises each article using LLMs, and rates them for factuality and political bias. Output is a daily Markdown report rendered in a Streamlit web UI.

---

## How it works

The agent runs a deterministic parallel pipeline — no LLM orchestration loop. Each run:

1. **search_news** — queries OpenRouter's Exa web search plugin, filtered to known-bias domains, to find recent articles per bias category
2. **extract_articles** — downloads and parses full article text via `newspaper4k`
3. **summarize_articles** — sends each article to a summarizer LLM (summary + key claims) and a rater LLM (factuality + bias scores); articles within a bias category are summarized in parallel
4. **compile_report** — assembles everything into a `BiasReport` per category, writes `output/YYYY-MM-DD.md` and returns a JSON summary

Steps 1–3 run concurrently across all bias categories via a `ThreadedPipelineRunner`. Four task-specific models are used to balance cost and quality:

| Task | Default model | Selectable in UI |
|---|---|---|
| Orchestrator | `google/gemini-2.5-pro` | Yes |
| Search | `perplexity/sonar` | Yes |
| Summarizer | `google/gemini-3.1-flash-lite-preview` | Yes |
| Rater | `anthropic/claude-sonnet-4.6` | Yes |

All models are routed through [OpenRouter](https://openrouter.ai).

---

## Project structure

```
spectrum-news-agent/
├── config.yml               # Single source of truth — models, sources, biases, prompts, output settings
├── app.py                   # Streamlit web UI entry point
├── src/
│   ├── agent.py             # NewsAgent — orchestrates the tool-call loop
│   ├── agent_models.py      # AgentContext, AgentResult, Context dataclasses
│   ├── article_extractor.py # newspaper4k wrapper with graceful degradation
│   ├── config.py            # OmegaConf loader with env var injection
│   ├── llm_service.py       # Multi-model LLM routing (OpenRouter / OpenAI SDK)
│   ├── llm_validation.py    # Safe JSON parsing of LLM responses
│   ├── logging_config.py    # JSON-structured logging with session/run context vars
│   ├── report_compiler.py   # Markdown + JSON report assembly
│   ├── schemas.py           # All data model dataclasses
│   ├── pipeline.py          # Pipeline abstraction — BiasProcessorProtocol, ThreadedPipelineRunner, NewsBiasPipeline
│   ├── sources.py           # Data-driven source registry (loaded from config.yml)
│   ├── event_store/         # SQLite monitoring backend (Protocol + Null Object)
│   └── tools/               # Tool definitions + handlers (one file per tool)
│       ├── search_news.py
│       ├── extract_articles.py
│       ├── summarize_articles.py
│       └── compile_report.py
└── tests/                   # pytest test suite (191 tests, ≥90% coverage)
```

---

## Requirements

- Docker (recommended) — or Python 3.11 + [Poetry](https://python-poetry.org/)
- An [OpenRouter](https://openrouter.ai) API key

---

## Quickstart

### 1. Set your API key

```bash
echo "OPENROUTER_API_KEY=sk-or-..." > .secrets
```

### 2. Web UI (Docker — recommended)

```bash
docker compose up --build web
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

- **Main window** — enter topics and select bias categories to include, then click **Run Analysis**. Results render in-page as a formatted report.
- **Sidebar** — optional settings: articles-per-category count, and a **Model Selection** expander with a dropdown per task (orchestrator, search, summarizer, rater) to choose from curated model options.

### 3. Run locally

```bash
poetry install
export OPENROUTER_API_KEY=sk-or-...
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501).

---

## Output

Each run saves a file to `output/YYYY-MM-DD.md` containing:

- Per-bias sections with all collected articles
- Per-article: source name + URL, MBFC factuality label, LLM factuality rating (high/mixed/low) + confidence, LLM bias rating + confidence, summary, key claims, and collapsed full article text

A JSON summary of the same data is returned in-process (and can be found in the compiled report result).

---

## Configuration

All behaviour is controlled by `config.yml`. No code changes are needed to:

- **Add a source** — add an entry under `sources:` with `name`, `domain`, `bias`, `factuality`
- **Add a bias category** — add an entry under `biases:` with `id` and `label`
- **Change a model** — update `openrouter.models.<task>.id`
- **Adjust article count** — set `agent.articles_per_bias`
- **Change output directory** — set `agent.output_dir`

Factuality labels follow [Media Bias / Fact Check (MBFC)](https://mediabiasfactcheck.com/) ratings: `very_high`, `high`, `mostly_factual`, `mixed`, `low`, `very_low`.

---

## Running tests

```bash
docker compose run --build --rm test
```

The suite runs 202 tests and enforces ≥90% code coverage.

---

## Monitoring

When `event_store.backend: sqlite` is set in `config.yml`, every run writes structured records to a local SQLite database (`data/events.db` by default):

- **events** — run lifecycle events (start, finish, errors)
- **llm_calls** — every LLM call with model, token usage, latency, status
- **tool_calls** — every tool invocation
- **article_metadata** — per-article ratings + extraction outcomes

Set `event_store.backend: null` to disable.
