#streamlit run psyckq_prototype.py
import streamlit as st
from textblob import TextBlob
from datetime import datetime
import json, os
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import io

# ---------- Config ----------
DATA_FILE = "psyckq_log.json"
MAX_HISTORY = 1000  # limit loaded history for performance

st.set_page_config(page_title="Psyck-Q Prototype", layout="wide")
# ---------- Task Tracker Config ----------
TASK_FILE = "psyckq_tasks.json"

def load_tasks():
    if os.path.exists(TASK_FILE):
        with open(TASK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_tasks(tasks):
    with open(TASK_FILE, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

if "tasks" not in st.session_state:
    st.session_state.tasks = load_tasks()

# ---------- Sidebar: Task Tracker ----------
with st.sidebar:
    st.header("📌 Task & Habit Tracker")

    with st.expander("Add New Task"):
        task_title = st.text_input("Task Title")
        task_reason = st.text_area("Why is this task important?")
        if st.button("Add Task"):
            if task_title.strip():
                st.session_state.tasks.append({
                    "title": task_title,
                    "reason": task_reason,
                    "status": "pending",
                    "comments": ""
                })
                save_tasks(st.session_state.tasks)
                st.success("Task added!")
            else:
                st.warning("Please enter a task title.")

    st.markdown("---")
    st.subheader("Pending Tasks")
    for idx, task in enumerate(st.session_state.tasks):
        if task["status"] == "pending":
            st.write(f"**{task['title']}** — {task['reason']}")
            if st.button(f"Mark as Done ✅ {idx}"):
                task["status"] = "done"
                save_tasks(st.session_state.tasks)
            comment = st.text_input(f"Comment on '{task['title']}'", key=f"comment_{idx}")
            if st.button(f"Save Comment 💬 {idx}"):
                task["comments"] = comment
                save_tasks(st.session_state.tasks)

    st.markdown("---")
    st.subheader("Completed Tasks")
    for task in st.session_state.tasks:
        if task["status"] == "done":
            st.write(f"**{task['title']}** ✅")
            st.caption(f"Reason: {task['reason']}")
            if task["comments"]:
                st.caption(f"Comment: {task['comments']}")

# ---------- Utility functions ----------
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                st.error("Error: Could not read log file. Starting fresh.")
                return []

            data = data[-MAX_HISTORY:]  # keep last N entries

            # 🔹 Upgrade or repair old entries
            fixed_data = []
            for entry in data:
                if not isinstance(entry, dict):
                    # Convert string/other types into a proper entry
                    entry = {
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "text": str(entry),
                        "analysis": analyze_text(str(entry)),
                        "private": True
                    }

                if "analysis" not in entry:
                    entry["analysis"] = analyze_text(entry.get("text", ""))

                if "date" not in entry:
                    entry["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                fixed_data.append(entry)

            return fixed_data
    return []

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def analyze_text(text):
    blob = TextBlob(text)
    sentiment = round(blob.sentiment.polarity, 3)
    tone = "Positive" if sentiment > 0.1 else "Negative" if sentiment < -0.1 else "Neutral"
    # simple distortion detection via keywords
    distortion_keywords = ["never", "always", "fail", "failure", "hate", "stuck", "can't", "cannot", "no one", "nobody", "worst", "useless"]
    words = [w.lower().strip(".,!?;:") for w in text.split()]
    distortions = list({w for w in words if w in distortion_keywords})
    # simple key phrase extraction: top nouns/adjectives via word frequency excluding stopwords
    freq = {}
    stopwords = set(["the","and","a","to","is","in","it","of","for","on","that","this","i","i'm","i've","you","your"])
    for w in words:
        if w and len(w) > 2 and w not in stopwords:
            freq[w] = freq.get(w,0) + 1
    top_phrases = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]
    top_phrases = [p[0] for p in top_phrases]
    return {"sentiment": sentiment, "tone": tone, "distortions": distortions, "key_phrases": top_phrases}

# CBT question library
CBT_QUESTIONS = {
    "catastrophizing": [
        "What's the worst that could realistically happen, and how would you cope?",
        "Is there evidence that challenges this thought?"
    ],
    "overgeneralization": [
        "Can you think of a specific example where this wasn't true?",
        "Is it fair to generalize from one event to everything?"
    ],
    "self_criticism": [
        "What would you say to a friend in the same situation?",
        "Can you identify something you did well recently?"
    ],
    "default": [
        "What would make you feel even slightly better right now?",
        "What's one small, manageable step you can take today?"
    ]
}

# mapping keywords to distortion type
DISTORTION_MAP = {
    "never": "overgeneralization",
    "always": "overgeneralization",
    "fail": "catastrophizing",
    "failure": "catastrophizing",
    "hate": "self_criticism",
    "stuck": "self_criticism",
    "can't": "catastrophizing",
    "cannot": "catastrophizing",
    "no one": "overgeneralization",
    "nobody": "overgeneralization",
    "worst": "catastrophizing",
    "useless": "self_criticism"
}

def determine_distortion_type(distortions):
    types = []
    for d in distortions:
        t = DISTORTION_MAP.get(d, None)
        if t and t not in types:
            types.append(t)
    if not types:
        return ["default"]
    return types

def generate_reframe(original_text, user_answers):
    # Very simple rule-based reframe combining user answers
    positives = []
    for a in user_answers:
        if a and len(a.strip())>0:
            positives.append(a.strip().capitalize())
    if positives:
        reframe = "Balanced thought: " + " ".join(positives)
    else:
        # fallback reframe
        reframe = "Balanced thought: This is one moment and not the whole story. Small steps can change things."
    return reframe

# ---------- Data load & session ----------
if 'data' not in st.session_state:
    st.session_state.data = load_data()

st.title("Psyck-Q  (Prototype)")
st.markdown("A lightweight CBT journaling assistant. All data stored locally in `psyckq_log.json`.")


# Sidebar: quick actions & team view
with st.sidebar:
    st.header("Controls")
    name = st.text_input("Your Name (optional)", value=st.session_state.data[-1]['meta'].get('name') if st.session_state.data else "")
    if st.button("Clear All Local Data"):
        if os.path.exists(DATA_FILE):
            os.remove(DATA_FILE)
        st.session_state.data = []
        st.success("Local data cleared. Refresh the page.")

    st.markdown("---")
    st.header("Connection Mode (Team View)")
    st.markdown("Toggle to see what a team/HR member would see. This view only shows aggregated metrics and no raw text.")
    show_team_view = st.checkbox("Show Team/HR View", value=False)

    st.markdown("---")
    st.header("Export")

    if st.button("Export JSON"):
        save_data(st.session_state.data)
        st.success(f"Saved {len(st.session_state.data)} entries to {DATA_FILE}")

    if st.button("Download Weekly Summary"):
        df = pd.DataFrame(st.session_state.data)
        recent = df.tail(7)

        # Build improved text summary
        summary = f"📅 Psyck-Q Weekly Summary\nDate: {datetime.now().date()}\nEntries Logged: {len(df)}\n\n"
        summary += "📊 Sentiment Breakdown:\n"
        tone_counts = recent['analysis'].apply(lambda x: x.get("tone", "Unknown") if isinstance(x, dict) else "Unknown").value_counts()
        for tone, count in tone_counts.items():
            summary += f"- {tone}: {count}\n"

        summary += "\n📝 Recent Entries:\n"
        for _, e in recent.iterrows():
            date_str = e.get("date", "Unknown Date")
            tone = e.get("analysis", {}).get("tone", "Unknown") if isinstance(e.get("analysis"), dict) else "Unknown"
            summary += f"- {date_str} — {tone}\n"

        # Save summary text to memory
        summary_bytes = io.BytesIO(summary.encode("utf-8"))

        # Generate chart
        plt.figure(figsize=(5,3))
        tones = [e.get("tone", "Unknown") if isinstance(e, dict) else "Unknown" for e in recent["analysis"]]
        plt.bar(range(len(tones)), [1]*len(tones), tick_label=tones)
        plt.title("Weekly Sentiment Trend")
        plt.ylabel("Entries")
        chart_bytes = io.BytesIO()
        plt.savefig(chart_bytes, format='png')
        plt.close()
        chart_bytes.seek(0)

        # Create ZIP in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zipf:
            zipf.writestr("psyckq_weekly_summary.txt", summary)
            zipf.writestr("weekly_chart.png", chart_bytes.getvalue())

        zip_buffer.seek(0)
        st.download_button(
            label="Download Weekly Summary (ZIP)",
            data=zip_buffer,
            file_name="psyckq_weekly_report.zip",
            mime="application/zip"
        )

    st.markdown("---")
    st.markdown("⚠️ This prototype is not a substitute for professional help. If you are in crisis, seek immediate assistance.")

# ---------- Main layout ----------
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("1. Journal / Chatbox")
    journal_text = st.text_area("Write your journal entry or vent here:", height=200)
    private = st.checkbox("Mark this entry PRIVATE (won't be shared in Connection Mode)", value=True)
    if st.button("Submit Entry"):
        if not journal_text.strip():
            st.warning("Please write something before submitting.")
        else:
            analysis = analyze_text(journal_text)
            # determine distortion types
            distortion_types = determine_distortion_type(analysis['distortions'])
            # pick first distortion type for prompts
            chosen_type = distortion_types[0] if distortion_types else "default"
            prompts = CBT_QUESTIONS.get(chosen_type, CBT_QUESTIONS['default'])
            # build entry skeleton and push to session state, then ask CBT prompts
            entry = {
                "meta": {"name": name or ""},
                "text": journal_text,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "analysis": analysis,
                "distortion_types": distortion_types,
                "private": bool(private),
                "cbt": {"prompts": prompts, "answers": [], "reframe": ""}
            }
            st.session_state.current_entry = entry
            st.session_state.show_cbt = True
            st.rerun()
    st.markdown("---")

    # CBT flow (appear after submit)
    if st.session_state.get("show_cbt", False) and st.session_state.get("current_entry", None):
        st.subheader("2. CBT Guided Prompts")
        entry = st.session_state.current_entry
        st.markdown(f"**Detected tone:** {entry['analysis']['tone']}  |  **Detected distortions:** {', '.join(entry['analysis']['distortions']) if entry['analysis']['distortions'] else 'None'}")
        answers = []
        for i, p in enumerate(entry['cbt']['prompts']):
            a = st.text_input(f"Q{i+1}: {p}", key=f"q_{i}", value="")
            answers.append(a)
        if st.button("Submit CBT Responses"):
            # save answers and create reframe
            entry['cbt']['answers'] = answers
            entry['cbt']['reframe'] = generate_reframe(entry['text'], answers)
            # append to data and save
            st.session_state.data.append(entry)
            save_data(st.session_state.data)
            # clear current cbt session
            st.session_state.current_entry = None
            st.session_state.show_cbt = False
            st.success("Entry saved with CBT responses. You can view it in Recent Entries below.")
            st.rerun()


    st.markdown("---")
    st.subheader("3. Recent Entries (Original → Reframe)")
    # show recent entries reversed
    for e in reversed(st.session_state.data[-20:]):
        with st.expander(f"{e['date']} — Tone: {e['analysis']['tone']} — {'PRIVATE' if e.get('private',False) else 'SHAREABLE'}"):
            st.markdown("**Original:**")
            st.write(e['text'])
            if e['cbt']['reframe']:
                st.markdown("**Reframe / Balanced Thought:**")
                st.write(e['cbt']['reframe'])
            else:
                st.markdown("*No reframe recorded.*")
            st.markdown("---")
            st.write("Analysis:")
            st.write(e['analysis'])
            if not e.get('private', False):
                st.info("This entry is shareable in Connection Mode.")

with col2:
    st.subheader("Dashboard & Connection Mode")
    df = pd.DataFrame(st.session_state.data)
    if df.empty:
        st.info("No entries yet. Your dashboard will appear here after you submit entries.")
    else:
        # mood chart for last 30 entries
        last_n = min(30, len(df))
        chart_df = df[-last_n:]
        sentiments = [d.get('sentiment', 0) if isinstance(d, dict) else 0 for d in chart_df['analysis']]
        dates = list(chart_df['date'])
        # small chart using matplotlib
        fig, ax = plt.subplots(figsize=(4,3))
        ax.plot(range(len(sentiments)), sentiments, marker='o')
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
        ax.set_title("Mood Trend (last entries)")
        ax.set_xlabel("Entry index (recent → latest)")
        ax.set_ylabel("Sentiment (-1 to 1)")
        st.pyplot(fig)

        # Distortion counts
        all_dist = []
        for a in df['analysis']:
            all_dist.extend(a.get('distortions', []))
        dist_counts = pd.Series(all_dist).value_counts() if all_dist else pd.Series([])
        st.markdown("**Common Cognitive Distortion Words**")
        if not dist_counts.empty:
            st.bar_chart(dist_counts)
        else:
            st.write("No distortion keywords detected yet.")

        # Connection Mode summary (aggregate shareable data only)
        shareable = [d for d in st.session_state.data if not d.get('private', True)]
        total = len(st.session_state.data)
        shareable_count = len(shareable)
        negative_count = sum(1 for d in shareable[-7:] if d['analysis']['tone']=="Negative") if shareable else 0
        avg_sentiment = round(sum(d['analysis']['sentiment'] for d in shareable)/shareable_count,3) if shareable_count else None
        st.markdown("**Connection Mode (Shareable Summary)**")
        st.write(f"Total entries: {total}  |  Shareable: {shareable_count}")
        if shareable_count:
            st.write(f"Avg sentiment (shareable): {avg_sentiment}")
            st.write(f"Negative entries in last 7 shareable: {negative_count}")
            conn_status = "⚠️ Needs attention" if negative_count >= 3 else "✅ Stable"
            st.info(f"Connection Mode Status: {conn_status}")
        else:
            st.write("No shareable entries yet. All entries are private by default.")

        # Team/HR view (no raw text)
        if show_team_view:
            st.markdown("---")
            st.subheader("Team / HR View (Aggregated Only)")
            st.write(f"User: {name or '[Anonymous]'}")
            st.write(f"Shareable entries: {shareable_count}")
            if shareable_count:
                st.write(f"Avg sentiment: {avg_sentiment}")
                st.write(f"Recent negative count: {negative_count}")
            st.write("Notes: This view contains aggregated metrics only. No raw journals are displayed.")

# ---------- End of app ----------
