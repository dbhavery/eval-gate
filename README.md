# eval-gate

An **evaluation release gate for LLM and RAG systems**. It runs a suite of graded
cases against a document corpus, checks each answer with real assertion,
retrieval, and grounding logic, and returns a **non-zero exit code when quality
regresses** against a recorded baseline, so a bad prompt, model, or retrieval
change fails CI the same way a broken unit test does.

This is a portfolio proof and a reusable gate, not a hosted product. It is
honest about its scope: the default mode is fully deterministic and offline, and
the "quality" metrics are explicit rule-based heuristics (documented below), not
an LLM judge.

## Start here (for reviewers)

Two commands show the whole point. A gate that passes clean, then fails on regression:

    pip install -e ".[dev]"
    python -m eval_gate run --suite fixtures/suite.yaml                                                # GATE PASSED, exit 0
    python -m eval_gate run --suite fixtures/suite_regressed.yaml --baseline fixtures/baseline.json    # GATE FAILED, exit 1

Then read `tests/` (97 tests, including the one that asserts the non-zero exit on
regression) and open the generated `reports/report.html`.

### The test that checks the tests

A suite that stays green after you break the code proves nothing, so there is a
third command:

    pytest -q tests/test_mutation.py    # 6 deliberate defects, each must turn the suite red

It copies the package, breaks one line, runs the suite, and requires a non-zero
exit, once per defect. The defects are real: the ones the 2026-09-19 audit found
here, plus the obvious next ones. It carries a control that runs an unmutated
copy and requires a clean pass, because every other assertion in it is of the
form "the suite failed", which a permanently broken harness would also satisfy.

On its first run one mutation survived. Disabling the pass-rate comparison
against the baseline left all 89 tests green, so nothing verified the sentence
this README leads with. `test_d13` in `tests/test_audit_2026_09_19.py` closes it.

`CALIBRATION.md` states what this suite has and has not been measured against,
including the fact that no grader here has been calibrated against human
judgment. Read it before reading a green gate as a quality claim.

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
- **One failing case fails the gate.** The pass-rate threshold is a second bar and
  cannot excuse a failing case. A suite that declares a check the tool cannot run,
  a check with no way to fail, or a case with no checks is refused at load time
  with exit `2` rather than scored.

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

# Gate a run against it. Passes because nothing regressed:
python -m eval_gate run --suite fixtures/suite.yaml --baseline fixtures/baseline.json
echo "exit: $?"   # 0
```

## Prove the gate fails on a regression

`fixtures/suite_regressed.yaml` points at deliberately-broken answers (a wrong
refund window, a policy contradiction, and a citation to a non-existent doc):

```bash
python -m eval_gate run --suite fixtures/suite_regressed.yaml --baseline fixtures/baseline.json
echo "exit: $?"   # 1  GATE FAILED (failing cases, threshold, baseline regression)
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
pytest -q            # 89 tests
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
| `eval_gate/gate.py` | Failing cases, threshold, baseline regression and suite weakening to a verdict + exit code |
| `eval_gate/report.py` | JSON report + self-contained dark-theme HTML |
| `eval_gate/cli.py` | `run` / `baseline` / `list-checks` |

### Check families

- **Assertions** (`checks/assertions.py`): `must_contain`, `must_not_contain`,
  `regex`, `json_schema` (a JSON-Schema subset validator), `refusal` (detects
  abstention and asserts it was expected or not).
- **Retrieval & grounding** (`checks/retrieval.py`): `recall_at_k`,
  `precision_at_k`, `mrr`, `ndcg_at_k`, and `citations_grounded`. Every
  `[doc:<id>]` the answer cites must be a real corpus document *and* one that was
  actually retrieved.
- **Quality** (`checks/quality.py`): `length` bounds, `forbidden_phrases`,
  `format` (json / markdown_list / plain), and `faithfulness`.

### Honest limitations

- **Faithfulness is a polarity-aware lexical-overlap heuristic**, not semantic
  entailment. It measures the fraction of the answer's content words that appear
  in the retrieved context with the same polarity, so a dropped or invented
  negation lowers the score. It catches off-topic hallucination, fabricated
  specifics, and a flipped claim. It will not catch a fluent paraphrase that
  changes meaning while reusing the context's words. Swap in an NLI model or LLM
  judge behind the same check interface for stronger signal.
- **No grader here is calibrated against human judgment**, and the suite holds no
  adversarial probes or canaries. `CALIBRATION.md` records what that means and
  what closing it would take.
- **`json_schema` validates a documented subset of JSON Schema** and refuses a
  schema using a keyword it cannot enforce, rather than ignoring the keyword and
  reporting a pass.
- **The TF-IDF retriever is a baseline** so the demo runs with zero setup. Point
  `runner` at your production retriever (same `retrieve(query, k) -> ids`
  signature) for real evaluation.
- Recorded fixture answers are exactly that: recorded once. Live-provider mode is
  where you evaluate a model under change.

## License

MIT. See [LICENSE](LICENSE).
