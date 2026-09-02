"""
Meeting Action-Item Agent
-------------------------
Takes a raw meeting transcript and extracts structured action items
(owner, task, deadline, confidence). Never "creates" a task directly —
every extraction must be explicitly confirmed by a human before it is
written to the task log. This is the core human-in-the-loop boundary
described in the job application: the agent proposes, a person approves.

Design notes (kept intentionally simple, on purpose):
- One model call, one job. No multi-agent chaining.
- Structured output enforced with a JSON schema + validation, not
  free-form prose parsing.
- Every run is logged (input hash, output, latency, token usage,
  validation result) so failures can be reproduced and measured.
- A stopping/confirmation gate: nothing is written to storage until
  the user clicks "Approve".
"""

import json
import os
import time
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

import anthropic

logging.basicConfig(
    filename=os.path.join(os.path.dirname(__file__), "agent_runs.log"),
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)

MODEL_NAME = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are an action-item extraction agent for meeting transcripts.

Your ONLY job is to read a transcript and extract concrete action items.
An action item must have a clear task. Owner and deadline are optional —
if the transcript does not state them, return null, do NOT guess.

Rules:
- Do not invent owners, deadlines, or tasks that are not supported by the text.
- If a sentence is vague ("we should look into this sometime") and has no
  clear owner or committed task, do NOT include it as an action item.
- Include a confidence score (0.0-1.0) per item, reflecting how explicit
  the transcript was about that item.
- Return ONLY valid JSON matching this schema, nothing else:

{
  "action_items": [
    {
      "task": "string, required",
      "owner": "string or null",
      "deadline": "string or null",
      "confidence": 0.0
    }
  ],
  "unclear_mentions": ["string, ...  brief notes on things that sounded like they might be action items but were too vague to extract"]
}
"""


@dataclass
class RunResult:
    success: bool
    action_items: list = field(default_factory=list)
    unclear_mentions: list = field(default_factory=list)
    latency_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None
    raw_response: Optional[str] = None


def _validate_schema(data: dict) -> Optional[str]:
    """Returns an error string if invalid, else None."""
    if "action_items" not in data or not isinstance(data["action_items"], list):
        return "missing or invalid 'action_items' list"
    for i, item in enumerate(data["action_items"]):
        if "task" not in item or not isinstance(item["task"], str) or not item["task"].strip():
            return f"item {i} missing required 'task' string"
        if "confidence" not in item or not isinstance(item["confidence"], (int, float)):
            return f"item {i} missing numeric 'confidence'"
        if not (0.0 <= float(item["confidence"]) <= 1.0):
            return f"item {i} confidence out of range"
    if "unclear_mentions" in data and not isinstance(data["unclear_mentions"], list):
        return "'unclear_mentions' must be a list"
    return None


def extract_action_items(transcript: str, client: Optional[anthropic.Anthropic] = None) -> RunResult:
    """Calls the model once, validates the output, logs the run.
    Never writes anything to persistent storage — that only happens
    after explicit human approval (see save_approved_items)."""

    if not transcript or not transcript.strip():
        return RunResult(success=False, error="empty transcript")

    if client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return RunResult(success=False, error="ANTHROPIC_API_KEY not set")
        client = anthropic.Anthropic(api_key=api_key)

    input_hash = hashlib.sha256(transcript.encode("utf-8")).hexdigest()[:12]
    start = time.time()

    try:
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": transcript}],
        )
    except Exception as e:
        logging.info(json.dumps({"input_hash": input_hash, "success": False, "error": f"api_error: {e}"}))
        return RunResult(success=False, error=f"API error: {e}")

    latency = time.time() - start
    raw_text = "".join(block.text for block in response.content if block.type == "text")

    try:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        data = json.loads(cleaned)
    except (json.JSONDecodeError, IndexError) as e:
        logging.info(json.dumps({
            "input_hash": input_hash, "success": False,
            "error": f"json_parse_error: {e}", "latency": latency,
        }))
        return RunResult(success=False, error=f"Model did not return valid JSON: {e}",
                          raw_response=raw_text, latency_seconds=latency)

    schema_error = _validate_schema(data)
    if schema_error:
        logging.info(json.dumps({
            "input_hash": input_hash, "success": False,
            "error": f"schema_error: {schema_error}", "latency": latency,
        }))
        return RunResult(success=False, error=f"Schema validation failed: {schema_error}",
                          raw_response=raw_text, latency_seconds=latency)

    result = RunResult(
        success=True,
        action_items=data["action_items"],
        unclear_mentions=data.get("unclear_mentions", []),
        latency_seconds=latency,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        raw_response=raw_text,
    )

    logging.info(json.dumps({
        "input_hash": input_hash,
        "success": True,
        "num_items": len(result.action_items),
        "latency": latency,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
    }))

    return result


def save_approved_items(items: list, log_path: str = "approved_tasks.jsonl") -> int:
    """The ONLY function allowed to persist action items, and only ever
    called after a human has explicitly approved each item in the UI.
    This is the confirmation gate — the agent proposes, it never commits
    on its own."""
    full_path = os.path.join(os.path.dirname(__file__), log_path)
    with open(full_path, "a") as f:
        for item in items:
            record = dict(item)
            record["approved_at"] = datetime.utcnow().isoformat()
            f.write(json.dumps(record) + "\n")
    return len(items)
