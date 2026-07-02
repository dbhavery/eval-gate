# eval-gate

An **evaluation release gate for LLM and RAG systems**. It runs a suite of graded
cases against a document corpus, checks each answer with real assertion,
retrieval, and grounding logic, and returns a **non-zero exit code when quality
regresses** against a recorded baseline — so a bad prompt, model, or retrieval
change fails CI the same way a broken unit test does.

This is a portfolio proof and a reusable gate, not a hosted product. It is
honest about its scope: the default mode is fully deterministic and offline, and
the "quality" metrics are explicit rule-based heuristics (documented below), not
an LLM judge.

## Start here (for reviewers)

Two commands show the whole point — a gate that passes clean, then fails on regression:

    pip install -e ".[dev]"
    python -m eval_gate run --suite fixtures/suite.yaml                                                # GATE PASSED, exit 0
    python -m eval_gate run --suite fixtures/suite_regressed.yaml --baseline fixtures/baseline.json    # GATE FAILED, exit 1

Then read `tests/` (54 tests, including the one that asserts the non-zero exit on regression) and open the generated `reports/report.html`.

## Why it exists

Most teams ship prompt and RAG changes with no automated quality gate. `eval-gate`
makes "did this change make the assistant worse?" a yes/no question a CI job can
answer, with a per-case HTML report a human can read in 30 seconds.

## What makes it trustworthy to run

- **Deterministic by default.** No API keys, no network. Answers come from
  recorded fixtures; retrieval is computed locally with a pure-Python TF-IDF
  retriever. `python -m eval_gate run` produces the same result on every machine.
- **Real checks, not stubs.** Assertions, IR metric math (recall/precision/MRR/
  NDCG), citation grounding, JSON-schema validity, and lexical faithfulness are
  all implemented and unit-tested.
- **A gate, not just a report.** It compares against a baseline and exits `1` on
  regression (`2` on a tool/config error).

## Install

```bash
git clone <this-repo> eval-gate
cd eval-gate
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Only dependency for the core is PyYAML. `[dev]` adds pytest.

## Run a passing evaluation (offline, deterministic)

```bash
python -m eval_gate run --suite fixtures/suite.yaml
```

Exits `0`, prints a per-case summary, and writes `reports/report.json` +
`reports/report.html`.

## Run the gate against a baseline

```bash
# Record a baseline from a known-good run (already checked in as fixtures/baseline.json):
python -m eval_gate baseline --suite fixtures/suite.yaml --out fixtures/baseline.json

# Gate a run against it — passes because nothing regressed:
python -m eval_gate run --suite fixtures/suite.yaml --baseline fixtures/baseline.json
echo "exit: $?"   # 0
```

## Prove the gate fails on a regression

`fixtures/suite_regressed.yaml` points at deliberately-broken answers (a wrong
refund window, a policy contradiction, and a citation to a non-existent doc):

```bash
python -m eval_gate run --suite fixtures/suite_regressed.yaml --baseline fixtures/baseline.json
echo "exit: $?"   # 1  — GATE FAILED (threshold + baseline regression)
```

## Open the HTML report

```bash
python -m eval_gate run --suite fixtures/suite.yaml
# then open reports/report.html
xdg-open reports/report.html      # Linux
open reports/report.html          # macOS
start reports/report.html         # Windows
```

## Run the tests

```bash
pytest -q            # 54 tests
```

## Optional: run against a live model

Provider mode is enabled only when the matching API key is set as an environment
variable. Keys are never read from files or flags.

```bash
export OPENAI_API_KEY=sk-...      # or ANTHROPIC_API_KEY=...
pip install -e ".[openai]"        # or .[anthropic]
python -m eval_gate run --suite fixtures/suite.yaml --provider openai
```

Retrieval is still computed locally; only the answer generation calls the model,
using the retrieved corpus documents as context (temperature 0). If the key is
unset, `--provider auto` falls back to deterministic mode.

## Architecture (60-second read)

```
suite.yaml ─┐
corpus/     ├─► runner ─► for each case:
responses/ ─┘                1. TfidfRetriever.retrieve(question)   → retrieved doc ids  (always local)
                             2. provider.answer(...)                → answer text        (fixture or live LLM)
                             3. run each check                      → CheckOutcome[]
                                       │
                                       ▼
                                  SuiteResult ─► gate.evaluate ─► GateVerdict (+ exit code)
                                       │                              │
                                       └────────► report (JSON + HTML) ◄──┘
```

| Module | Responsibility |
|---|---|
| `eval_gate/config.py` | Typed suite schema + YAML loader/validator |
| `eval_gate/corpus.py` | Corpus loading + deterministic TF-IDF retriever |
| `eval_gate/providers.py` | Deterministic fixture provider; optional OpenAI/Anthropic |
| `eval_gate/checks/` | The check registry and three families (below) |
| `eval_gate/runner.py` | Orchestrates a suite → `SuiteResult` |
| `eval_gate/gate.py` | Threshold + baseline regression → verdict + exit code |
| `eval_gate/report.py` | JSON report + self-contained dark-theme HTML |
| `eval_gate/cli.py` | `run` / `baseline` / `list-checks` |

### Check families

- **Assertions** (`checks/assertions.py`): `must_contain`, `must_not_contain`,
  `regex`, `json_schema` (a JSON-Schema subset validator), `refusal` (detects
  abstention and asserts it was expected or not).
- **Retrieval & grounding** (`checks/retrieval.py`): `recall_at_k`,
  `precision_at_k`, `mrr`, `ndcg_at_k`, and `citations_grounded` — every
  `[doc:<id>]` the answer cites must be a real corpus document *and* one that was
  actually retrieved.
- **Quality** (`checks/quality.py`): `length` bounds, `forbidden_phrases`,
  `format` (json / markdown_list / plain), and `faithfulness`.

### Honest limitations

- **Faithfulness is a lexical-overlap heuristic**, not semantic entailment. It
  measures the fraction of the answer's content words that appear in the
  retrieved context. It catches off-topic hallucination and fabricated specifics;
  it will not catch a fluent paraphrase that subtly changes meaning. Swap in an
  NLI model or LLM judge behind the same check interface for stronger signal.
- **The TF-IDF retriever is a baseline** so the demo runs with zero setup. Point
  `runner` at your production retriever (same `retrieve(query, k) -> ids`
  signature) for real evaluation.
- Recorded fixture answers are exactly that — recorded once. Live-provider mode
  is where you evaluate a model under change.

## License

MIT. See [LICENSE](LICENSE).
