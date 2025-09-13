import streamlit as st
from datetime import datetime
from utils import speech_utils, db_utils, tts_utils
import sqlite3
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords
import re
from openai import OpenAI
from transformers import pipeline

# ------------------- NLTK setup -------------------
nltk.download("stopwords", quiet=True)

# ------------------- OpenAI Setup -------------------
OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------- Database Init -------------------
conn = db_utils.init_db()

# ------------------- Safe Rerun -------------------
def safe_rerun():
    """Safely rerun Streamlit without crashing."""
    try:
        st.experimental_rerun()
    except RuntimeError:
        pass  # Ignore Streamlit internal runtime errors


# ------------------- Streamlit Page Setup -------------------
st.set_page_config(page_title="🎙️ Voice Notes App", layout="wide")
st.title("🎙️ Voice-Controlled Notes App")
st.write("Speak your notes, save them, and manage easily!")

# ===================== Notes =====================
try:
    notes = db_utils.get_notes(conn)
except sqlite3.OperationalError as e:
    st.error(f"Error fetching notes: {e}")
    notes = []

# --- Dashboard Summary ---
st.subheader("📊 Notes Summary")
if notes:
    total_notes = len(notes)
    last_note = notes[-1]
    last_note_text = last_note[1][:50] + "..." if len(last_note[1]) > 50 else last_note[1]
    last_note_time = last_note[2] if last_note[2] else "Unknown Date"

    col1, col2, col3 = st.columns(3)
    col1.metric("📝 Total Notes", total_notes)
    col2.metric("🕒 Last Note Time", last_note_time)
    col3.metric("✍️ Last Note Preview", last_note_text)
else:
    st.info("No notes yet. Start recording your first note!")

# --- Record & Save Note ---
if st.button("🎙️ Record & Save Note"):
    text = speech_utils.recognize_speech()
    if text:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        db_utils.add_note(conn, text, ts)
        st.success(f"✅ Note saved: {text}")
        safe_rerun()
    else:
        st.error("❌ Could not recognize speech.")

# --- Search Notes ---
st.subheader("🔍 Search Notes")
query = st.text_input("Enter a keyword to search:")
filtered_notes = notes
if query:
    filtered_notes = [n for n in notes if query.lower() in n[1].lower()]
    st.write(f"Found {len(filtered_notes)} result(s) for **{query}**")

# --- Export Notes ---
st.subheader("📤 Export Notes")
if filtered_notes:
    # TXT
    txt_buffer = io.StringIO()
    for n in filtered_notes:
        note_time = n[2] if n[2] else "Unknown Date"
        txt_buffer.write(f"{n[0]} | {n[1]} | {note_time}\n")
    st.download_button("📄 Download TXT", txt_buffer.getvalue(), file_name="voice_notes.txt")

    # PDF
    pdf_buffer = io.BytesIO()
    pdf = canvas.Canvas(pdf_buffer, pagesize=letter)
    width, height = letter
    y = height - 40
    pdf.setFont("Helvetica", 12)
    pdf.drawString(30, y, "Voice Notes Export")
    y -= 30
    for n in filtered_notes:
        note_time = n[2] if n[2] else "Unknown Date"
        line = f"{n[0]} | {n[1]} | {note_time}"
        pdf.drawString(30, y, line)
        y -= 20
        if y < 40:
            pdf.showPage()
            pdf.setFont("Helvetica", 12)
            y = height - 40
    pdf.save()
    pdf_buffer.seek(0)
    st.download_button("📑 Download PDF", pdf_buffer, file_name="voice_notes.pdf")
else:
    st.info("No notes to export.")

# --- Display Notes ---
st.subheader("📝 Your Notes")
if filtered_notes:
    for n in filtered_notes:
        note_time = n[2] if n[2] else "Unknown Date"
        with st.expander(f"Note {n[0]} ({note_time})"):
            st.write(n[1])
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"🗑️ Delete {n[0]}", key=f"del{n[0]}"):
                    db_utils.delete_note(conn, n[0])
                    safe_rerun()
            with col2:
                if st.button(f"🔊 Play {n[0]}", key=f"play{n[0]}"):
                    tts_utils.speak(n[1])
else:
    st.info("No notes found." if query else "No notes saved yet.")

# ===================== Analytics =====================
if notes:
    st.subheader("📈 Notes Analytics")
    df = pd.DataFrame(notes, columns=["id", "text", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
    df["date"] = df["timestamp"].dt.date

    # Notes per Day Chart
    st.write("### 📅 Notes per Day")
    notes_per_day = df.groupby("date").size()
    fig, ax = plt.subplots()
    notes_per_day.plot(kind="bar", ax=ax)
    ax.set_ylabel("Number of Notes")
    ax.set_xlabel("Date")
    st.pyplot(fig)

    # Word Cloud
    st.write("### ☁️ Word Cloud")
    all_text = " ".join(df["text"].tolist())
    if all_text.strip():
        wc = WordCloud(width=800, height=400, background_color="white").generate(all_text)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.info("Not enough text for word cloud.")

# ===================== AI / Keyword Summary =====================
if notes:
    st.subheader("🧠 Smart Weekly Summary")
    all_text = " ".join([n[1] for n in notes])
    summary_type = st.radio(
        "Choose summary method:",
        ("Keyword Summary (Fast, Offline)", "AI Summary (OpenAI GPT / Hugging Face)")
    )

    summary = ""
    if summary_type == "Keyword Summary (Fast, Offline)":
        stop_words = set(stopwords.words("english"))
        words = re.findall(r"\w+", all_text.lower())
        words = [w for w in words if w not in stop_words and len(w) > 2]
        word_freq = Counter(words).most_common(5)
        if word_freq:
            keywords = [w for w,_ in word_freq]
            summary = f"This week, your notes mostly talk about: **{', '.join(keywords)}**."
            st.success(summary)
        else:
            st.info("Not enough data for keyword summary yet.")

    elif summary_type == "AI Summary (OpenAI GPT / Hugging Face)":
        if len(all_text) > 50:
            backend = st.selectbox("Select AI Backend:", ["OpenAI GPT", "Hugging Face BART"])
            try:
                if backend == "OpenAI GPT":
                    prompt = f"Summarize the following notes into a short weekly report (3-4 sentences):\n\n{all_text}"
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=120,
                        temperature=0.7
                    )
                    summary = response.choices[0].message.content.strip()
                    st.success(summary)
                elif backend == "Hugging Face BART":
                    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                    summary_obj = summarizer(all_text, max_length=100, min_length=30, do_sample=False)
                    summary = summary_obj[0]["summary_text"]
                    st.success(summary)
            except Exception as e:
                st.error(f"AI summary failed: {e}")
        else:
            st.info("Not enough data for AI summary yet.")

    # --- Speak & Download Summary ---
    if summary:
        if st.button("🔊 Speak Summary"):
            tts_utils.speak(summary)

        if st.button("💾 Download as MP3"):
            filepath = tts_utils.save_as_mp3(summary, "weekly_summary.mp3")
            with open(filepath, "rb") as f:
                st.download_button("⬇️ Download Weekly Summary (MP3)", f, file_name="weekly_summary.mp3", mime="audio/mpeg")

        st.download_button("📄 Download Weekly Summary (TXT)", summary, file_name="weekly_summary.txt", mime="text/plain")

        if st.button("💾 Save Summary to Database"):
            db_utils.save_summary(summary)
            safe_rerun()

# ===================== Saved Summaries =====================
st.write("### 📜 Saved Weekly Summaries")
try:
    saved_summaries = db_utils.fetch_summaries()
except sqlite3.OperationalError as e:
    st.error(f"Error fetching summaries: {e}")
    saved_summaries = []

if saved_summaries:
    for s in saved_summaries:
        summary_time = s[2] if s[2] else "Unknown Date"
        with st.expander(f"Summary {s[0]} ({summary_time})"):
            st.write(s[1])
            if st.button(f"🗑️ Delete Summary {s[0]}", key=f"del_summary_{s[0]}"):
                db_utils.delete_summary(s[0])
                safe_rerun()
else:
    st.info("No saved summaries yet.")