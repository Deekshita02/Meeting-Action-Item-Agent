Meeting Action-Item Agent

A small, single-purpose agent that reads a meeting transcript and extracts action items (owner, task, deadline, confidence) — with a hard human-approval gate before anything is saved. Built to be small enough to fully explain, not to look impressive.

Problem statement

Meeting notes generate action items that often get lost or misassigned. This agent proposes structured action items from a transcript; a person reviews and approves each one individually before it's persisted anywhere. The agent never takes an autonomous "write" action.

Architecture

Built as a LangGraph workflow:

extractor -> grounding_check -> END                 (clean input: fast path)
                   |
                   +-> reviewer -> END              (suspicious input)
                          |
                          +-> back to extractor with feedback (max 3 attempts)
extractor: the original single Gemini call with JSON schema validation (unchanged)
grounding_check: free, rule-based check. It flags owners/deadlines that do not appear in the transcript and injection-style wording
reviewer: independent LLM call with its own prompt. It runs only when grounding_check flags something, so clean transcripts skip the extra call
human approval gate: unchanged. Nothing is saved until a person approves each item individually
Transcript (text)
      │
      ▼
[ Streamlit UI ]  ──calls──▶  [ graph_agent.py: run_graph() ]
                                     │
                                     ▼
                              extractor node
                         (Gemini call, JSON schema validation)
                                     │
                                     ▼
                           grounding_check node
                        (rule-based, no API call)
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                   ▼
               clean → END                      flagged → reviewer node
                                                 (independent Gemini call)
                                                          │
                                        ┌─────────────────┴─────────────────┐
                                        ▼                                     ▼
                                  issues found                          approved
                              → back to extractor                          → END
                              with feedback (max 3x)
                                     │
                                     ▼
                     shown to user for per-item review
                                     │
                       user checks "Approve" on individual items
                                     │
                                     ▼
                           save_approved_items()
                           → only function that writes
                             to approved_tasks.jsonl

Every run (success or failure) is logged to agent_runs.log with an input hash, latency, token counts, and outcome — so a failure can be looked up and reproduced without storing full transcript text in the log.

Setup
bash
git clone <this-repo>
cd Meeting-Action-Item-Agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in GOOGLE_API_KEY (get one free at aistudio.google.com/apikey)
export GOOGLE_API_KEY=your-key-here   # or use python-dotenv / Streamlit secrets
streamlit run app.py

To deploy publicly (e.g. for a shareable link), push this repo to GitHub and deploy on Streamlit Community Cloud, setting GOOGLE_API_KEY as a secret in the app settings.

Sample input/output

Input:

Priya: Raj, can you send the revised pricing deck by Thursday?
Raj: Sure, Thursday works.
Priya: Someone should really look into the churn numbers at some point.
Meera: I'll take the churn analysis, first pass by next Friday.

Output:

json
{
  "action_items": [
    {"task": "Send revised pricing deck", "owner": "Raj", "deadline": "Thursday", "confidence": 0.95},
    {"task": "First pass on churn analysis", "owner": "Meera", "deadline": "next Friday", "confidence": 0.9}
  ],
  "unclear_mentions": ["someone should look into churn numbers - no owner or commitment stated"]
}

Note the vague "someone should look into churn" line is correctly not turned into a fabricated action item with an invented owner. This case is clean, so it never reaches the reviewer node and returns at the extractor's normal latency.

Evaluation

test_cases.json has 12 cases covering: a normal case, an ambiguous case with no clear owner, a missing-deadline case, a hedged-commitment case, a misleading-date case, a similar-names case, a case with conflicting/updated deadlines, an empty-input edge case, and two prompt-injection attempts embedded in transcript text.

Run it:

bash
python eval_runner.py

Latest run (Gemini 3.5 Flash Lite, 12 cases):

Version	Checks passed	Cases passed	Avg latency
Single agent	36/36	12/12	0.59s
LangGraph + reviewer	36/36	12/12	0.79s and 0.82s (two runs)

An earlier LangGraph run averaged 3.6s during a Gemini slowdown, so latency varies with API load. The earlier numbers in this README (19/19 on 7 cases, 2.32s) were measured on a different model (Gemini 3.5 Flash) and an earlier, smaller test set — they are not directly comparable to the table above.

Both versions pass every case on this test set, so this does not show an accuracy gain from adding the reviewer. What it does show is an independent check that runs on suspicious inputs at a small average latency cost — on a larger or messier set of real transcripts, that's where the reviewer would be expected to catch cases the extractor alone would miss.

This is deliberately a small, honest eval set, meant to show the methodology — automated pass/fail per check, categorized by failure type, aggregate pass rate and latency — not a claim of exhaustive coverage. In a real deployment this would grow to 30-50+ cases and add human-reviewed scoring for anything subjective (e.g., "is this task description reasonable phrasing"), while objective checks (schema validity, null-vs-hallucinated fields, injection resistance) stay automated.

What requires human review vs. automated scoring, in my approach:

Automated: JSON schema validity, presence/absence of owner & deadline fields, item counts within bounds, injection resistance, owner/deadline grounding against the source transcript.
Human review: whether the phrasing of an extracted task is actually useful/accurate — this is inherently judgment-based and doesn't reduce well to a single automatic metric, so I'd sample a percentage of live runs weekly for manual spot-checks rather than trying to fully automate it.
Decision log
Reviewer false rejections: the first reviewer prompt flagged correct output on a prompt-injection input (it listed the injected text as a problem even though the extractor had already ignored it), which cost an extra, unnecessary revision round. Fixed by separating the reviewer's reasoning from its issues list, restricting it to five defined issue types, and treating "no confirmed issues" as approved rather than defaulting to rejection.
Conditional review, not always-on: an always-on reviewer made every run slower, even for clean transcripts that didn't need a second check. A cheap, rule-based grounding check now decides when the LLM reviewer actually runs, so the common case stays fast.
Retries: temporary API errors (503) are retried with backoff. Quota errors (429) are not retried, since waiting a few seconds can't fix a daily limit — the run fails visibly instead of hanging.
Known limitations
The reviewer trigger is keyword and string based. A rephrased injection can bypass it — one test case's "system update" wording never reached the reviewer at all; the extractor happened to handle it correctly on its own, not because the trigger caught it.
The reviewer does not run on every input, so it can miss an invented task whose owner and deadline wording happen to already appear somewhere in the transcript.
The eval has 12 cases, and some checks are still weak: the injection check is written around one specific test case's exact text rather than the general pattern, and the similar-names case only counts items rather than checking which name got which task.
No retries for daily quota limits (429s fail immediately by design, see decision log above).
Single model provider — no fallback if Gemini has an outage.
Confidence score is model-reported, not independently calibrated against a labeled dataset — I'd want to validate it against human-labeled data before trusting it for auto-filtering.
No auth / multi-user isolation — fine for a portfolio demo, not for shipping with real company transcripts as-is.
What I'd improve next
Replace the keyword-based grounding check with something more robust to rephrased injection attempts.
Expand the eval set to 30-50 cases and track pass-rate over time as a regression signal when I change the prompt or model.
Add a second model provider as a fallback for outages.
Add a lightweight dedup step against previously approved items.
Move the approved-items log from a local JSONL file to a real database with per-user access control before using it with real data.
