# Meeting Action-Item Agent

A small, single-purpose agent that reads a meeting transcript and extracts
action items (owner, task, deadline, confidence) — with a hard human-approval
gate before anything is saved. Built to be small enough to fully explain,
not to look impressive.

## Problem statement

Meeting notes generate action items that often get lost or misassigned.
This agent proposes structured action items from a transcript; a person
reviews and approves each one individually before it's persisted anywhere.
The agent never takes an autonomous "write" action.

## Architecture

```
Transcript (text)
      │
      ▼
[ Streamlit UI ]  ──calls──▶  [ agent.py: extract_action_items() ]
                                     │
                                     ▼
                         Claude API (single call, system prompt
                         enforces JSON schema, no tools/loops)
                                     │
                                     ▼
                          JSON parse + schema validation
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                                ▼
              validation fails                  validation passes
              → shown as error,                 → shown to user for
                raw output visible                per-item review
                for debugging
                                                        │
                                          user checks "Approve"
                                          on individual items
                                                        │
                                                        ▼
                                        save_approved_items()
                                        → only function that writes
                                          to approved_tasks.jsonl
```

Every run (success or failure) is logged to `agent_runs.log` with an input
hash, latency, token counts, and outcome — so a failure can be looked up
and reproduced without storing full transcript text in the log.

## Setup

```bash
git clone <this-repo>
cd meeting-agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
export ANTHROPIC_API_KEY=your-key-here   # or use python-dotenv / Streamlit secrets
streamlit run app.py
```

To deploy publicly (e.g. for a shareable link), push this repo to GitHub and
deploy on [Streamlit Community Cloud](https://streamlit.io/cloud), setting
`ANTHROPIC_API_KEY` as a secret in the app settings.

## Sample input/output

Input:
```
Priya: Raj, can you send the revised pricing deck by Thursday?
Raj: Sure, Thursday works.
Priya: Someone should really look into the churn numbers at some point.
Meera: I'll take the churn analysis, first pass by next Friday.
```

Output:
```json
{
  "action_items": [
    {"task": "Send revised pricing deck", "owner": "Raj", "deadline": "Thursday", "confidence": 0.95},
    {"task": "First pass on churn analysis", "owner": "Meera", "deadline": "next Friday", "confidence": 0.9}
  ],
  "unclear_mentions": ["someone should look into churn numbers - no owner or commitment stated"]
}
```

Note the vague "someone should look into churn" line is correctly *not*
turned into a fabricated action item with an invented owner.

## Evaluation

`eval/test_cases.json` has 7 cases covering: a normal case, an ambiguous
case with no clear owner, a case with a missing deadline, an empty-input
edge case, a prompt-injection attempt embedded in transcript text, a
multi-item case, and a case with conflicting/updated deadlines.

Run it:
```bash
python eval/eval_runner.py
```

This is deliberately a *small, honest* eval set (7 cases), meant to show the
methodology — automated pass/fail per check, categorized by failure type,
aggregate pass rate and latency — not a claim of exhaustive coverage. In a
real deployment this would grow to 30-50+ cases and add human-reviewed
scoring for anything subjective (e.g., "is this task description reasonable
phrasing"), while objective checks (schema validity, null-vs-hallucinated
fields, injection resistance) stay automated.

**What requires human review vs. automated scoring, in my approach:**
- Automated: JSON schema validity, presence/absence of owner & deadline
  fields, item counts within bounds, injection resistance.
- Human review: whether the phrasing of an extracted task is actually
  useful/accurate — this is inherently judgment-based and doesn't reduce
  well to a single automatic metric, so I'd sample a percentage of live
  runs weekly for manual spot-checks rather than trying to fully automate it.

## Known limitations

- Single model call, no retries on transient API failures yet.
- No conversation memory across multiple transcripts / no dedup if the
  same action item appears in two meetings.
- Confidence score is model-reported, not independently calibrated against
  a labeled dataset — I'd want to validate it against human-labeled data
  before trusting it for auto-filtering.
- Prompt-injection defense here is basic (system/user separation +
  schema validation catches the obvious case in the eval set); a
  production version would need broader adversarial testing.
- No auth / multi-user isolation — fine for a portfolio demo, not for
  shipping with real company transcripts as-is.

## What I'd improve next

- Add retry/backoff and a circuit breaker for API failures.
- Expand the eval set to 30-50 cases and track pass-rate over time as a
  regression signal when I change the prompt or model.
- Add a lightweight dedup step against previously approved items.
- Move the approved-items log from a local JSONL file to a real database
  with per-user access control before using it with real data.
