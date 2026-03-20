# Evaluation Guide

Synapse includes a built-in evaluation framework for benchmarking models against standard and custom tasks. Evaluations run against a loaded model, measure accuracy or perplexity, and store results per-tenant in SQLite.

## Supported Benchmarks

| Benchmark    | CLI alias              | What it measures                        | Scoring method              |
|-------------|------------------------|-----------------------------------------|-----------------------------|
| `mmlu`       | `mmlu`                 | Multiple-choice accuracy (MMLU-style)   | Letter match (A/B/C/D)      |
| `hellaswag`  | `hellaswag`            | Commonsense reasoning completion        | Contains match              |
| `humaneval`  | `humaneval`, `human_eval` | Code generation (HumanEval-style)    | Contains match              |
| `perplexity` | `perplexity`, `ppl`    | Language modeling quality               | Approximate sliding-window  |
| `custom`     | `custom`               | User-defined prompt/expected pairs      | Contains match              |

All benchmarks read samples from a JSONL dataset file. Each line must be a JSON object with at least `prompt` and `expected` fields. MMLU additionally uses `choices` (array of strings) and `answer_index` (integer).

## CLI Usage

The CLI sends inference requests to the local Synapse API server at `http://127.0.0.1:8420/inference`, so the server must be running with a model loaded before you start an eval.

### Basic command

```bash
synapse eval --model <model-name> --benchmark <benchmark> --dataset <path-to-jsonl>
```

### Examples

Run MMLU on 100 samples:

```bash
synapse eval --model llama-3.1-8b --benchmark mmlu --dataset ./data/mmlu.jsonl --sample-limit 100
```

Run perplexity evaluation:

```bash
synapse eval --model mistral-7b --benchmark ppl --dataset ./data/wikitext.jsonl
```

Run a custom benchmark:

```bash
synapse eval --model llama-3.1-8b --benchmark custom --dataset ./data/my-eval.jsonl
```

### CLI output

For accuracy-based benchmarks (MMLU, HellaSwag, HumanEval, Custom):

```
Score: 0.8500 (85.0%)
Samples evaluated: 200
Duration: 42.50s
```

For perplexity:

```
Perplexity: 5.23 (lower is better)
Samples evaluated: 1000
Duration: 120.00s
```

## REST API Usage

All eval endpoints are tenant-scoped. The tenant is determined by the request's authentication context.

### Create a run

```
POST /eval/runs
```

Request body:

```json
{
  "model_name": "llama-3.1-8b",
  "benchmarks": ["mmlu", "hellaswag"],
  "sample_limit": 100,
  "dataset_path": "/data/eval.jsonl"
}
```

- `model_name` (required) -- name of the loaded model to evaluate.
- `benchmarks` (required) -- array of benchmark kinds: `perplexity`, `mmlu`, `hellaswag`, `human_eval`, `custom`.
- `sample_limit` (optional) -- cap the number of samples per benchmark.
- `dataset_path` (optional) -- path to JSONL dataset. When provided, the run executes in the background immediately.

Response (`201 Created`):

```json
{
  "run_id": "a1b2c3d4-...",
  "model_name": "llama-3.1-8b",
  "status": "queued",
  "benchmarks": ["mmlu", "hellaswag"],
  "results": [],
  "error": null
}
```

The run transitions through statuses: `queued` -> `running` -> `completed` (or `failed`).

### List runs

```
GET /eval/runs?page=1&per_page=20
```

Returns a paginated list of eval runs belonging to the current tenant.

### Get a specific run

```
GET /eval/runs/:id
```

Response:

```json
{
  "run_id": "a1b2c3d4-...",
  "model_name": "llama-3.1-8b",
  "status": "completed",
  "benchmarks": ["mmlu", "hellaswag"],
  "results": [
    {
      "benchmark": "mmlu",
      "score": 0.72,
      "samples_evaluated": 100,
      "duration_secs": 45.3,
      "evaluated_at": "2026-03-19T12:00:00Z"
    },
    {
      "benchmark": "hellaswag",
      "score": 0.65,
      "samples_evaluated": 100,
      "duration_secs": 38.1,
      "evaluated_at": "2026-03-19T12:01:00Z"
    }
  ],
  "error": null
}
```

If a run failed, `status` is `"failed"` and `error` contains the reason.

## Custom Benchmark Format

Create a JSONL file where each line is a JSON object:

```jsonl
{"prompt": "What is the capital of France?", "expected": "Paris"}
{"prompt": "What is 2 + 2?", "expected": "4"}
```

For MMLU-style benchmarks, include choices and the correct answer index:

```jsonl
{"prompt": "What is 2+2?", "expected": "A", "choices": ["4", "5", "6", "7"], "answer_index": 0}
```

Scoring uses **contains match** for most benchmarks: the model's output passes if it contains the expected string. MMLU uses **letter match**, comparing the model's output against the expected letter (A, B, C, or D).

## Result Interpretation

### Accuracy benchmarks (MMLU, HellaSwag, HumanEval, Custom)

- **Score range**: 0.0 to 1.0 (fraction of correct answers).
- A score of 0.85 means 85% of samples matched.
- The `details` field includes `total_samples`, `successful_inferences`, and `scoring_method`.

### Perplexity

- **Score range**: positive float, unbounded. Lower is better.
- Uses approximate sliding-window prediction. The text is split at the midpoint; the model is prompted with the first half and scored on how well its output matches the second half.
- Typical good values depend on the corpus and model size.

### Run statuses

| Status      | Meaning                                        |
|------------|------------------------------------------------|
| `queued`    | Run created, not yet started                   |
| `running`   | Benchmarks are being executed                  |
| `completed` | All benchmarks finished successfully           |
| `failed`    | A benchmark errored out; check the `error` field |

### Pass/fail judgment

Synapse reports raw scores without a built-in pass/fail threshold. Compare scores across models or against known baselines to determine acceptability for your use case.

## Multi-Tenant Evaluation

All evaluation data is tenant-scoped:

- **CLI**: runs under the `default` tenant.
- **REST API**: tenant is extracted from the request's authentication context (`TenantId` extension).
- **Storage**: the SQLite `eval_results` table includes a `tenant_id` column. Queries filter by tenant automatically.

Tenant isolation is enforced at every layer. A `GET /eval/runs/:id` request returns a 404 if the run belongs to a different tenant, even if the run ID is valid. Listing runs only returns runs owned by the requesting tenant.
