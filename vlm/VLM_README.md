# Claude odd-one-out study scaffold

This folder implements the foundation of the supplied 2,000-trial plan. Each trial sends three labeled images in one fresh Messages request, receives four JSON fields, validates the probabilities, and saves a result immediately. No conversation history passes between trials. Reported probabilities are **elicited judgments**, not established calibrated choice probabilities.

## Start offline

Run from the project root using the existing Python environment:

```powershell
.venv\Scripts\python.exe vlm\run_trials.py --mode check
.venv\Scripts\python.exe vlm\run_trials.py --mode mock --limit 10 --output runs/mock-pilot-v3
.venv\Scripts\python.exe -m unittest discover -s vlm -p "test_*.py" -v
```

`check` validates the complete manifest and image signatures, then constructs the selected requests without sending them. `mock` writes explicitly synthetic probabilities to exercise persistence. Neither requires the Anthropic SDK or a key. The ten demo triplets reuse six existing reference images; they are plumbing examples, not the research sample.

All relative command arguments resolve against `vlm/`. Paths inside a trial CSV resolve against that CSV's directory. Output paths must stay inside `vlm/`. Images elsewhere in the project are only read.

## Prepare the actual sample

Run the Python cells in `vlm/getting_vlm_batch.qmd` to generate `vlm/trials.csv`
from all rows of `data/probability_estimates/agg_5plus.csv`. The notebook maps
zero-based triplet IDs to CLIP row positions, checks the saved CLIP index, and
uses each `unique_id` to locate its `.jpg` reference image. It preserves source
row order and A/B/C ordering and validates every mapping before writing.

The generated manifest uses these columns:

```csv
trial_id,image_a,image_b,image_c,id_a,id_b,id_c
study-0001,../data/path/to/a.jpg,../data/path/to/b.jpg,../data/path/to/c.jpg,127,733,1106
```

The paths above only illustrate the format. Required columns are `trial_id,image_a,image_b,image_c`; the optional object IDs are retained for analysis. Trial IDs must be unique. A/B/C order is preserved, with no sorting or randomization. Use the same IDs and order as the human and embedding data, and join downstream on the full ordered ID triplet. The runner executes the manifest provided; it does not force a 2,000-row sample.

## Run the paid pilot when ready

First confirm Claude **API** credits and access with the account owner. A chat subscription alone does not establish API access. Set up dependencies in a dedicated environment inside this folder:

```powershell
py -m venv vlm\.venv
vlm\.venv\Scripts\python.exe -m pip install -r vlm\requirements.txt
Copy-Item vlm\config.example.json vlm\config.pilot.json
```

Edit `config.pilot.json` to select an available, explicitly versioned Claude vision model supporting structured output. The model is intentionally unset rather than choosing a research condition for you. Review whether that model accepts the configured temperature. Set `ANTHROPIC_API_KEY` through your normal secure environment mechanism; do not paste it into source or configuration files. Keys are never recorded by the runner.

`
`secureKey = Read-Host "Enter your Claude API key" -AsSecureString`
`$env:ANTHROPIC_API_KEY = [System.Net.NetworkCredential]::new("", $secureKey).Password`
`Remove-Variable secureKey`
`
Test with:
`Test-Path Env:ANTHROPIC_API_KEY`

`"temperature": null` omits the temperature parameter and uses the API's sampling behavior. This is required for the current SDK and Sonnet 5.5; do not replace it with `0` for that model. Numeric temperature is retained only for older SDK/model combinations that support it. Offline `check` now verifies request argument compatibility when the SDK is installed, without sending a request. This does not verify server-side model access or credits. See the [Sonnet 5.5 migration guide](https://platform.claude.com/docs/en/models/sonnet-5-5/migration-guide).

After updating from runner version 1, use a fresh output directory such as `runs/live-pilot-v2`. Old snapshots and failures remain available for review. Local `TypeError` exceptions now stop immediately instead of being retried across trials.

Runner version 3 uses `prob_a`, `prob_b`, and `prob_c` throughout the prompt,
structured response schema, validation, mock responses, and results CSV. Older
runs used `probability_a`, `probability_b`, and `probability_c`. Start a fresh
output directory when upgrading; changing an old CSV's headers alone does not
make its saved prompt, schema, or settings hash compatible. Historical run
snapshots and raw responses retain the original field names.

```powershell
vlm\.venv\Scripts\python.exe vlm\run_trials.py --trials trials.csv --config config.pilot.json --mode check
vlm\.venv\Scripts\python.exe vlm\run_trials.py --trials trials.csv --config config.pilot.json --mode live --limit 10 --output runs/live-pilot-v3
```

Inspect all ten explanations, A/B/C alignment, probability sums, failure logs, and token usage. Record the installed SDK with `vlm\.venv\Scripts\python.exe -m pip freeze` if archiving the environment. The run snapshot also records its installed version. Freeze the prompt/settings after pilot review. If anything changes, use a new run directory. A production run can use `--output runs/study-2000` without `--limit`; it repeats the pilot rows as part of a fresh frozen study. Alternatively, if every setting and input is unchanged, reuse the pilot directory without `--limit` to continue and retain its first ten results.

## Saved artifacts and recovery

- `settings.json`: config, full prompt, output schema, runner version, SDK version, ordered manifest, resolved image paths, and SHA-256 image hashes. Do not edit this snapshot.
- `results.csv`: validated successes, with original trial/object IDs, paths, `prob_a`, `prob_b`, `prob_c`, explanation, requested and returned model, settings hash, UTC time, mode, request ID, and input/output token usage.
- `responses.jsonl`: raw responses, including invalid, refused, or truncated replies, for inspection.
- `failures.jsonl`: failed attempts, error type, timestamp, and HTTP status where available.

Re-run the same command to skip successful trial IDs and retry unfinished trials. Changed images, prompt, config, SDK version, or manifest require a new output directory. Invalid distributions are flagged, never silently normalized. Each retry is another independent request; both invalid responses and transient failures can trigger up to `max_attempts` attempts per invocation. SDK retries are disabled so they do not compound this limit. Authentication and common permanent request errors stop the run immediately. Exhausted trials produce exit code 1; setup or fatal errors produce 2.

A directory lock prevents concurrent writers. If a process is forcibly killed, confirm it has stopped before removing that run's `.lock` file. Writes are incremental but are not transactional with the API: interruption after an API response but before CSV persistence can cause that trial to be sent again. Inspect saved responses when reconciling interrupted runs. A partially written/corrupt CSV causes resume to stop rather than guess. Retries can incur additional API cost and select for valid responses; retain the attempt logs for research reporting.

## Scope and next steps

Implemented: serial independent requests, fixed prompt and structured schema, offline preflight and mocks, probability validation, bounded retries, raw audit data, and guarded resume. A serial runner is sufficient to execute 2,000 trials, subject to API limits and runtime. Optional batch submission, pricing estimates, automated human/embedding comparisons, and pilot-quality judgments are future work. No live API pilot has been performed as part of scaffolding.

API references: [structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs), [Python Messages API](https://platform.claude.com/docs/en/api/python/messages/create). Numerical ranges, finite values, nonempty explanations, and sum-to-one constraints are checked locally in addition to the API schema.
