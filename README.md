# Programming run failure reproducibility

Reproducibility materials for **Predicting Run Failure Before Execution in Introductory Programming Education: Process Data and Local Language Models Across Two Cohorts** (IEEE Access manuscript Access-2026-51757, revised analysis).

The primary unit is a run event. Current compilation status, compilation history, and earlier execution outcomes are distinct information sources. The revised evaluation uses student-disjoint validation, complete nested stacking, paired student bootstrap intervals, and direct cross-year tests. This repository reports conditional predictive evidence; it does not establish educational effectiveness, a general model-size threshold, or information containment between code and process.

## Reproduce the released results

Use Python 3.12 and a clean virtual environment:

```sh
python -m venv .venv
# Activate .venv for your operating system, then:
python -m pip install -r requirements.txt
python revision/reanalyse.py --part all
```

The default command reuses the released held-out predictions and reproduces the bootstrap summaries. To regenerate fitted predictions, including the full nested fusion procedure:

```sh
python revision/reanalyse.py --part all --refit
```

No live LLM calls are needed for either command. Saved language-model scores remain fixed. Full refitting can take several minutes. Differences in libraries or CPU numerical behavior can change fitted predictions; versions are pinned below and the released run environment is in `revision/analysis/runtime.json`.

The released derived-data package was checked by recomputing all nine primary process, LLM, calibration, sensitivity and transfer summary JSON files; they matched the authoring analysis exactly. To recreate the four main vector figures, optionally install `matplotlib==3.11.2` and run `python revision/make_revision_figures.py`.

## Populations and samples

| Population | 2019 | 2021 |
|---|---:|---:|
| Original eligible labelled events | 29,457 | 17,371 |
| Primary events | 29,457 | 16,489 |
| Recorded-clean primary events | 20,887 | 14,354 |
| Independent all-runs LLM sample | 886 | 881 |
| Independent recorded-clean LLM sample | 859 | 887 |

The 2021 primary population excludes 882 events belonging to 441 ambiguous exact-match keys, retaining neither of two candidate buffer states for a single outcome. Earlier recorded outcomes are formed before this restriction and can include these events; the outcome-history-free ablation removes that dependency. Student and event identifiers are the datasets' public pseudonyms, not personal identities.

The original LLM sampler draws 1,000 events from each pool with `random.Random(20260919)`. The draws are independent samples, with overlaps of 22 and 69 identifiers. All multiply generated overlap events are excluded across models; 21 of the 22 2019 overlap events otherwise meet outcome eligibility. The stored `_arm` flag is not authoritative for overlap records because the original sampler shared mutable objects. `intended_all` and `intended_clean` in the event manifests reconstruct membership using the original order and seed. Do not merge the two samples and call their enriched union a representative all-runs sample.

## Files

- `data/analysis_input_*.csv`: numeric/indicator analysis features, labels, public identifiers, and membership needed to refit the revised analysis. Student source text, grades and execution output are omitted.
- `revision/audit/event_manifest_*.csv`: event-level original eligibility, sampling membership, parsing consistency, duplicate exclusions and source hashes.
- `corpus/llm_cache_*.jsonl`: archived generated responses, scores, answer labels and retained first-token evidence. Original duplicate responses are preserved. The field allow-list removes unused local notes and error fields; SHA-256 hashes of the original private local files are supplied separately for provenance.
- `revision/analysis/*predictions*.csv`: held-out classical and nested fusion predictions, with identifiers and outcomes.
- `revision/analysis/*.json`: bootstrap intervals, paired differences, calibration, sensitivity results, and runtime versions.
- `revision/analysis/repeat_check_responses.jsonl`: separate current repeatability audit, not replacements for historical evaluation scores. Ten prompts were requested twice per model. Prompt source text is not redistributed here.
- `source_preparation/`: original replay/outcome scripts, preserved for inspection. Their historical assumptions and limitations are described below; they are not a validated generic ProgSnap2 parser.

All assignment cells with both outcomes and at least five students enter the main LLM macro AUC, including small cells previously excluded by a 50-event threshold. `sensitivity_*.json` retains that historical threshold as an explicit sensitivity. The main process model fits separately within each assignment; sampled classical comparators fit across assignments with student-disjoint folds. Cross-year models fit only the source cohort. These training populations should not be interchanged.

## Raw source data and preparation

Source releases:

- John Edwards, **2019 CS1 Keystroke Data**, Harvard Dataverse, version 5.2: https://doi.org/10.7910/DVN/6BPCXN
- John Edwards, **2021 CS1 Keystroke Data**, Harvard Dataverse, version 7.0: https://doi.org/10.7910/DVN/BVOF7S

Download the official release files and consult their `readme.txt` and `sample-code.ipynb`. The latter already provides the edit-replacement reconstruction example: this repository extends it to run snapshots, linkage, analysis membership and model evaluation rather than claiming the operation is new. Both Dataverse releases specify CC0. This repository does not include student-grade tables (including the withdrawn 2019 table), raw source files or the 2.4 GB raw execution output.

For historical preparation, place the original files under `corpus/cs1_2019/` and `corpus/cs1/`, respectively. From `corpus/`, run the preserved scripts with their output paths understood:

```sh
python ../source_preparation/build_runset.py
python ../source_preparation/extract_runs2021.py
python ../source_preparation/build_runset2021.py
```

After obtaining those source runsets and installing the named local Ollama models, `python source_preparation/local_repeat_check.py` can repeat the separate diagnostic against localhost. It explicitly makes new requests and writes new audit files; it is not required to reproduce archived evaluation results. Original evaluation requests are not silently rerun by the main analysis.

These commands reproduce the historical exports, not the revised inclusion exclusions. The revised membership flags are retained explicitly in `revision/audit/` and `data/`. The 2021 join uses student, assignment, file and exact millisecond timestamp; it has no one-to-one guarantee. A replay's matching compilation status is only a binary consistency check, not proof that reconstructed and executed source are identical. Replay initialization at an empty buffer remains an assumption when no checkpoint exists.

## Scoring and validation limits

The Y/N prompt is:

````text
```python
{SOURCE}
```

Will this program raise an error when executed? Answer Y for yes, N for no. One character only.
````

The request has one user message, temperature zero, top-20 token log probabilities and a 4,000-token output budget. The configured 7B/14B/32B context is 8,192 tokens. Scores normalize the available Y and N token probabilities. An absent alternative is treated as saturation at zero or one; these are ranking scores, not established run-failure probabilities. The 7B/14B/32B caches retain first-token evidence, limited to eight alternatives and rounded log probabilities. The 3B caches predate evidence retention and contain continuous scores whose extraction route is not independently recoverable. They are an exploratory comparison, not verified text-fallback scores.

Five outer folds group students. Nested fusion generates inner student-held-out classical scores within each outer training partition and uses fixed L2 logistic regression (C=1). A logistic LLM-only control and training-reference rank averaging are also evaluated. No outer-test score selects the method. Paired bootstrap intervals resample students 2,000 times while keeping predictions fixed. They omit training, prompt, generation and task-selection variability; invalid class-degenerate replicates are counted.

Historical process counters differ between recorders. Prior outcome availability assumes the earlier run completed before the indexed run; completion times are insufficient to guarantee this for all overlaps. The no-outcome-history arm avoids prior-failure features. Missing local dependencies, input values, saved-file behavior and source drift remain limitations. Fresh repeated requests are a small diagnostic and do not replace or retroactively validate archived responses.

## Attribution and license

Analysis scripts added for this revision are offered under the MIT license in `LICENSE`. Derived features and original model-response records in this repository are provided under CC0; cite the source datasets and this study when using them. Original dataset rights and collection ethics remain documented in the Dataverse releases. The public materials contain no manuscript author biographies, private review correspondence, API keys, or unrelated workspace files.
