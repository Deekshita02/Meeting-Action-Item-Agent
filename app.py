import streamlit as st
from agent import extract_action_items, save_approved_items

st.set_page_config(page_title="Meeting Action-Item Agent", page_icon="✅", layout="centered")

st.title("✅ Meeting Action-Item Agent")
st.caption(
    "Paste a meeting transcript. The agent extracts action items (owner, task, "
    "deadline, confidence). Nothing is saved until you review and approve each item — "
    "the agent proposes, you decide."
)

if "result" not in st.session_state:
    st.session_state.result = None
if "approved_count" not in st.session_state:
    st.session_state.approved_count = 0

with st.expander("Try a sample transcript"):
    sample = (
        "Priya: Okay, let's wrap up. Raj, can you send the revised pricing deck "
        "by Thursday?\n"
        "Raj: Sure, Thursday works.\n"
        "Priya: Great. Also, someone should really look into the churn numbers "
        "at some point.\n"
        "Meera: I'll take the churn analysis, I can have a first pass by next Friday.\n"
        "Priya: Perfect. And I'll follow up with legal on the contract redlines "
        "this week.\n"
    )
    st.code(sample)
    if st.button("Use this sample"):
        st.session_state.transcript_input = sample

transcript = st.text_area(
    "Transcript",
    key="transcript_input",
    height=220,
    placeholder="Paste transcript text here...",
)

col1, col2 = st.columns([1, 3])
with col1:
    run_clicked = st.button("Extract action items", type="primary")

if run_clicked:
    if not transcript.strip():
        st.warning("Paste a transcript first.")
    else:
        with st.spinner("Extracting..."):
            st.session_state.result = extract_action_items(transcript)
            st.session_state.approved_count = 0

result = st.session_state.result

if result is not None:
    if not result.success:
        st.error(f"Extraction failed: {result.error}")
        if result.raw_response:
            with st.expander("Raw model output (for debugging)"):
                st.code(result.raw_response)
    else:
        st.success(
            f"Found {len(result.action_items)} candidate action item(s) "
            f"in {result.latency_seconds:.1f}s "
            f"({result.input_tokens} in / {result.output_tokens} out tokens)."
        )

        if result.unclear_mentions:
            with st.expander(f"⚠️ {len(result.unclear_mentions)} vague mention(s) not extracted"):
                for m in result.unclear_mentions:
                    st.write(f"- {m}")

        st.subheader("Review before approving")
        approved_items = []
        for i, item in enumerate(result.action_items):
            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.markdown(f"**{item['task']}**")
                    st.caption(
                        f"Owner: {item.get('owner') or '— not stated —'} | "
                        f"Deadline: {item.get('deadline') or '— not stated —'} | "
                        f"Confidence: {item.get('confidence', 0):.2f}"
                    )
                with c2:
                    approve = st.checkbox("Approve", key=f"approve_{i}")
                if approve:
                    approved_items.append(item)

        if st.button(f"Save {len(approved_items)} approved item(s)", disabled=len(approved_items) == 0):
            n = save_approved_items(approved_items)
            st.session_state.approved_count += n
            st.success(f"Saved {n} approved item(s) to the task log.")

if st.session_state.approved_count:
    st.info(f"Total approved this session: {st.session_state.approved_count}")

st.divider()
st.caption(
    "This is a deliberately small, single-purpose agent: one model call, one job, "
    "a validated schema, and a hard human-approval gate before anything persists. "
    "See eval/ for the evaluation harness and agent_runs.log for run traces."
)
