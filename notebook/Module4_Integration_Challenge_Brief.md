# Module 4 Integration Challenge: Claims & Provider Network Assistant

**Duration:** 60 minutes | **Format:** Solo or pairs | **Notebook:** `Module4_Integration_Challenge_Starter.ipynb`

## Scenario

You're building a **Claims & Provider Network Assistant** for a health plan member support line.
Members ask about their claim status or whether a provider is in-network. You must build the same
kind of production-grade pipeline you just walked through in the facilitator demo — but for this
new domain, from a blank starter.

**Do not copy the Benefits Assistant code verbatim.** The point is to apply the *pattern*, not
transcribe the *exact* code. Field names, functions, and data are different here on purpose.

## What "done" looks like

By the end of the hour, you should have a working `ClaimsAssistant` class such that:

```python
assistant = ClaimsAssistant()
assistant.ask("What is the status of claim CLM-1001?")
assistant.ask("Is provider NPI-2002 in network?")
assistant.ask("")  # should not crash
assistant.ask("What is the status of claim CLM-9999?")  # unknown claim, should not crash
```
...runs cleanly end-to-end for every question in the provided test bank (see notebook), and produces
readable log entries in `claims_log.jsonl`.

## Requirements checklist (self-review rubric — tick these off as you go)

- [ ] **Hardened call wrapper** — a `call_llm()` function with an explicit timeout, and retry with
  backoff on transient errors (`APITimeoutError`, `APIConnectionError`, `RateLimitError`). Does not
  retry on non-transient errors.
- [ ] **Structured output schema** — a system prompt that forces JSON-only output with named fields,
  and a schema dict describing required fields + types.
- [ ] **Validator** — a `validate_response()` function that checks: (a) all required fields present,
  (b) correct types, (c) at least one *business sanity check* (e.g. an approved amount can't be negative).
- [ ] **Repair or fallback on invalid output** — if validation fails, either re-prompt with the error,
  or fall back to a safe, honest template. Never pass unvalidated data through.
- [ ] **Per-stage error handling** — the pipeline does not crash on: empty input, an unknown claim/provider
  ID, or a simulated API failure. Each failure returns a sensible fallback, not a stack trace.
- [ ] **One non-agentic function dispatch** — the model is given exactly one closed-list function to call
  (either `get_claim_status` or `check_provider_network`, your choice) via `tools`/`tool_choice`, and your
  code executes at most one function per question, followed by exactly one follow-up call for the final answer.
  No loops, no chained multi-function calls.
- [ ] **The model correctly abstains** — test a question that shouldn't trigger your function, and confirm
  it doesn't get called.
- [ ] **Logging** — every `ask()` call writes structured JSON-line log entries (input, result, correlation_id,
  latency) to `claims_log.jsonl`, using a single correlation ID per question.
- [ ] **Assembled class** — all of the above lives inside one `ClaimsAssistant` class with a single
  public `.ask(question)` method that never raises to the caller.

## Suggested pacing (self-paced checkpoints — not enforced, just a sanity guide)

| Time | You should be roughly here |
|---|---|
| 0–10 min | Schema + system prompt drafted; mock data reviewed |
| 10–25 min | `call_llm()` wrapper + `validate_response()` working |
| 25–40 min | Error handling in place; function dispatch wired for one function |
| 40–50 min | Logging added; everything assembled into `ClaimsAssistant` |
| 50–60 min | Run the full test bank; fix what breaks; tick off the rubric |

## Mock data and test questions are provided in the starter notebook

You do not need to invent claim/provider data or test questions — they're given, so you can focus your
time on the pipeline logic itself.

## If you get stuck

Raise your hand — don't spend more than ~5 minutes stuck on one thing. The facilitator has a reference
solution and will circulate throughout.
