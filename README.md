# eval-gate

## The problem

A team changes a prompt, swaps a model, or adjusts retrieval, and the assistant
starts giving wrong answers or citing documents that do not exist. No unit test
fails, because unit tests do not read the answers. Nobody notices until
customers do.

## How it solves it

`eval-gate` runs a fixed set of questions against a document set and checks
each answer with these checks:

1. **Text rules.** Required and banned phrases, regular expressions, JSON
   structure, length, output format, and whether the answer refused when it
   should (or should not) have.
2. **Retrieval scores.** Recall, precision, MRR and NDCG at k for the documents
   the retriever returned.
3. **Citations.** Every `[doc:<id>]` the answer cites must be a real document in
   the set and one that was actually retrieved for that question.
4. **Faithfulness.** The share of the answer's content words that appear in the
   retrieved text with the same polarity, so a dropped or added "not" lowers
   the score.

It then compares the run to a saved baseline and exits non-zero when quality
drops. The gate fails when any of these is true:

- any case failed (a pass-rate threshold cannot excuse a failing case),
- the pass rate is below the suite's threshold,
- a case that passed in the baseline now fails, or the pass rate fell,
- the suite itself was weakened: a case lost checks the baseline ran, or the
  threshold was lowered.

A suite that names a check the tool cannot run, a check with no way to fail, or
a case with no checks is refused when it loads, with exit code `2`, instead of
being scored.

## Who it helps

- Teams shipping a RAG or support assistant who want a pass or fail in CI when
  a prompt, model or retrieval change makes answers worse.
- Leads and reviewers who want a per-case report showing which answers changed
  and why they failed.

## Technical details

- **Language and dependencies.** Python 3.11 or later. The core needs only
  PyYAML. OpenAI and Anthropic clients are optional extras.
- **Offline by default.** With no API key, answers come from recorded fixtures
  and retrieval runs locally with a pure-Python TF-IDF retriever, so the same
  suite gives the same result on every machine.
- **Exit codes.** `0` gate passed, `1` gate failed, `2` tool or config error.
- **Reports.** Each run writes `reports/report.json` and a self-contained
  `reports/report.html` unless `--no-report` is passed.
- **Tests.** The tests check each assertion, the retrieval metric math,
  citation grounding, faithfulness, config loading and refusal of bad suites,
  the gate rules above, the report writer and the CLI, including an end-to-end
  run that requires a non-zero exit on the regressed suite.
- **Mutation tests.** `tests/test_mutation.py` copies the package, breaks one
  line at a time (for example, making `must_contain` always pass or ignoring the
  baseline comparison), and requires the test suite to fail for each one. A
  control run on an unmodified copy must pass, so a broken harness cannot report
  success.
- **CI.** GitHub Actions on Python 3.11, 3.12 and 3.13 runs the tests, the
  mutation tests, the gate on the good suite, and a step that fails the build
  if the gate does not fail on the regressed suite.
- **Calibration.** `CALIBRATION.md` states what the suite has and has not been
  measured against. No grader here has been calibrated against human judgment.

## Install

```bash
git clone https://github.com/dbhavery/eval-gate
cd eval-gate
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

`[dev]` adds pytest.

## Quick start

A gate that passes, then the same gate failing on a regression:

```bash
python -m eval_gate run --suite fixtures/suite.yaml --baseline fixtures/baseline.json             # GATE PASSED, exit 0
python -m eval_gate run --suite fixtures/suite_regressed.yaml --baseline fixtures/baseline.json   # GATE FAILED, exit 1
```

`fixtures/suite_regressed.yaml` points at deliberately broken answers: a wrong
refund window, a policy contradiction, and a citation to a document that does
not exist.

## Run an evaluation

```bash
python -m eval_gate run --suite fixtures/suite.yaml
```

Prints a per-case summary and writes `reports/report.json` and
`reports/report.html`.

## Record and use a baseline

```bash
# Record a baseline from a known-good run (one is checked in as fixtures/baseline.json):
python -m eval_gate baseline --suite fixtures/suite.yaml --out fixtures/baseline.json

# Gate a run against it:
python -m eval_gate run --suite fixtures/suite.yaml --baseline fixtures/baseline.json
echo "exit: $?"
```

## List the available checks

```bash
python -m eval_gate list-checks
```

## Run the tests

```bash
pytest -q --ignore=tests/test_mutation.py    # unit and end-to-end tests
pytest -q tests/test_mutation.py             # mutation tests
```

## Optional: run against a live model

Provider mode is enabled only when the matching API key is set as an environment
variable. Keys are never read from files or flags.

```bash
export OPENAI_API_KEY=sk-...      # or ANTHROPIC_API_KEY=...
pip install -e ".[openai]"        # or .[anthropic]
python -m eval_gate run --suite fixtures/suite.yaml --provider openai
```

Retrieval is still computed locally. Only answer generation calls the model,
with the retrieved documents as context and temperature 0. With
`--provider auto`, the tool uses Anthropic if `ANTHROPIC_API_KEY` is set, then
OpenAI if `OPENAI_API_KEY` is set, and otherwise the recorded fixtures.

## Architecture

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
| `eval_gate/config.py` | Typed suite schema, YAML loader and validator |
| `eval_gate/corpus.py` | Corpus loading and the TF-IDF retriever |
| `eval_gate/providers.py` | Fixture provider; optional OpenAI and Anthropic providers |
| `eval_gate/checks/` | The check registry and check families listed below |
| `eval_gate/runner.py` | Runs a suite and builds a `SuiteResult` |
| `eval_gate/gate.py` | Applies the gate rules and sets the verdict and exit code |
| `eval_gate/report.py` | JSON report and self-contained HTML report |
| `eval_gate/cli.py` | `run`, `baseline` and `list-checks` commands |

### Check families

- **Assertions** (`checks/assertions.py`): `must_contain`, `must_not_contain`,
  `regex`, `json_schema` (a JSON Schema subset validator), `refusal` (detects
  abstention and asserts it was expected or not).
- **Retrieval and grounding** (`checks/retrieval.py`): `recall_at_k`,
  `precision_at_k`, `mrr`, `ndcg_at_k`, and `citations_grounded`.
- **Quality** (`checks/quality.py`): `length` bounds, `forbidden_phrases`,
  `format` (json, markdown_list or plain), and `faithfulness`.

### Limitations

- **Faithfulness is a lexical-overlap heuristic**, not semantic entailment. It
  catches off-topic answers, made-up specifics and a flipped claim. It will not
  catch a fluent paraphrase that changes the meaning while reusing the source's
  words. An NLI model or LLM judge can be placed behind the same check
  interface for a stronger signal.
- **No grader is calibrated against human judgment**, and the suite holds no
  adversarial probes or canaries. `CALIBRATION.md` describes what closing that
  gap would take.
- **`json_schema` validates a documented subset of JSON Schema.** A schema that
  uses a keyword outside that subset is refused rather than partly checked.
- **The TF-IDF retriever is a baseline** so the demo runs with no setup. Point
  the runner at your retriever (same `retrieve(query, k) -> ids`
  signature) for a real evaluation.
- Recorded fixture answers were recorded once. Live-provider mode is where you
  evaluate a model that is changing.

## License

MIT. See [LICENSE](LICENSE).
