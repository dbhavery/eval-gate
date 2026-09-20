# Calibration status

This file records what the `eval-gate` suite has been measured against and what
it has not. It is written for a reviewer who wants to know how much weight a
green gate carries here. Last updated 2026-09-19.

## What the suite contains

| Item | Count |
|---|---|
| Cases in `fixtures/suite.yaml` | 13 |
| Check instances across those cases | 48 |
| Corpus documents | 6 (616 words total) |
| Recorded answer fixtures | 13 good, 13 in the regressed set |
| Graders registered in the check registry | 14 |
| Graders the suite actually uses | 10 |

The four things the 48 checks measure:

1. Retrieval rank (`recall_at_k`, 12 instances).
2. Citation validity (`citations_grounded`, 11 instances).
3. Factual content of the answer (`must_contain` 13, `faithfulness` 4,
   `must_not_contain` 1, `forbidden_phrases` 1, `refusal` 1).
4. Output shape (`format` 2, `length` 2, `json_schema` 1).

`mrr`, `ndcg_at_k`, `precision_at_k` and `regex` are implemented and unit-tested
but no case uses them. They are library code, not suite coverage.

## What the suite does not contain

**There are zero adversarial probes and zero canaries in this repo.** No case
tries to defeat a grader, no case is a known-bad answer planted to confirm a
grader still fires during a real run, and nothing in the suite detects a grader
that has silently stopped working. The `tests/` directory covers grader logic in
isolation; that is unit testing, not a probe of the running gate.

**Calibration against human judgment: none.**

- No case carries a human label. `relevant_ids` is the author's own judgment of
  which document answers the question, written at the same time as the question.
- No second rater has scored any case, so there is no inter-rater agreement
  figure of any kind (no Cohen's kappa, no Krippendorff's alpha, no percent
  agreement).
- No held-out set of human-scored answers exists, so no grader in this repo has
  a measured false-pass or false-fail rate. When `faithfulness` returns 0.62,
  nothing here establishes what a human would have said about the same answer.
- The three regressed fixtures (`refund_window`, `warranty_length`,
  `privacy_data_sale`) were written by the same person who wrote the graders, for
  the purpose of being caught. They demonstrate that the gate fires on defects it
  was built to find. That is self-consistency. It is not calibration, and it
  cannot surface a defect class the author did not think of.

**No threshold in this repo has a recorded justification.** All seven commits on
`master` are silent on how any number was chosen. Specifically:

| Number | Where | Basis on record |
|---|---|---|
| `threshold: 0.85` | both suite files, until 2026-09-19 | none |
| `faithfulness min: 0.8` | 4 cases | none |
| `max_words: 60` | 2 cases | none |
| `recall_at_k min: 1.0` | 12 cases | none |

The 2026-09-19 audit found that `threshold: 0.85` was not a neutral choice. With
13 cases it permitted one entirely wrong answer to pass: 12/13 = 0.923 cleared
0.85, and the gate exited 0 on a run where the refund window was off by 60 days,
the answer cited nothing, and faithfulness scored 0.600. The threshold is now
1.0, and a failing case fails the gate regardless of the pass rate. The reason
for 1.0: every case in this suite is written as a requirement the system must
meet, and no case carries a severity label, so there is no principled way to
decide which failures are tolerable. Until cases carry severity, the only
defensible bar is all of them.

`faithfulness min: 0.8` and `max_words: 60` remain unjustified numbers. They are
plausible, they have not been measured, and this file says so rather than
implying otherwise.

## What the graders can and cannot see

- `faithfulness` compares content words between the answer and the retrieved
  context, and since 2026-09-19 it compares the polarity of the clause each word
  sits in, so dropping or adding a negation changes the score. It is still
  lexical. A fluent paraphrase that changes a number's meaning without changing
  the words can pass it. It is not an entailment model.
- `must_contain` matches a substring. "Refunds are not available within 30 days"
  satisfies `must_contain: "30 days"`. Pairing the check with `faithfulness` and
  `must_not_contain` reduces this exposure; it does not close it.
- `citations_grounded` confirms a cited document exists and was retrieved. It
  does not confirm that the cited document supports the sentence it is attached
  to.
- `recall_at_k` and the other IR metrics score the local TF-IDF retriever, which
  exists so the demo runs offline. They say nothing about a production retriever.

## What it would take to calibrate this

Ordered by how much each step would change the confidence a green gate deserves.

1. **Human labels on the existing 13 cases.** For each case, a second person
   writes the correct answer and the set of documents that support it, without
   seeing the author's version. Disagreements are resolved and recorded. This
   turns `relevant_ids` from one person's opinion into an agreed ground truth.
2. **A scored answer set, separate from the suite.** 100 to 200 answers to the
   same questions, some correct, some wrong in specific ways (wrong number,
   right number with a dropped negation, fluent paraphrase with a changed
   meaning, correct content with a fabricated citation, refusal where an answer
   existed). Two raters score each one pass or fail. Agreement is computed and
   reported. This set is the instrument that measures the graders.
3. **A measured operating point for every threshold.** Run each grader across
   that scored set and report its false-pass and false-fail rate at several
   thresholds. `faithfulness min` is then chosen from the curve, with the number
   and the rate it buys recorded next to it. `max_words: 60` is either derived
   from the length distribution of answers humans judged acceptable, or dropped.
4. **Adversarial probes, as cases.** One case per grader whose answer is built to
   defeat that grader while staying wrong: a negation-flipped answer for
   `faithfulness`, a substring-satisfying but false answer for `must_contain`, a
   real-but-irrelevant citation for `citations_grounded`. Each probe asserts the
   grader fails it.
5. **Canaries inside the running gate.** A small set of cases whose answers are
   known bad, run on every invocation, which must fail. If a canary passes, a
   grader has broken and the gate reports a tool error rather than a pass. A gate
   that cannot demonstrate a failure proves nothing about a run where everything
   passed.
6. **Drift checks on the corpus.** The recorded answers are correct with respect
   to a fixed 6-document corpus. If the corpus changes, every expectation needs
   re-confirming. Nothing currently detects that.

Until steps 1 through 3 exist, the honest reading of a green gate here is: the
answers still contain the strings, the citations still resolve, the retrieval
still ranks the labelled document in the top 3, and nothing regressed against the
recorded baseline. That is a real regression gate. It is not evidence that the
graders agree with a human on what a good answer is.
