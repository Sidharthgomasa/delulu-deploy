import streamlit as st
import pandas as pd
import re
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
from textblob import TextBlob

st.set_page_config(page_title="😵‍💫 Delulu Chat Analyzer", page_icon="😵‍💫", layout="wide")

# ------------------ GEN Z CSS ------------------
st.markdown("""
<style>
body { background-color: #0f0f0f; color: white; }
.big-title { font-size: 48px; font-weight: 900; text-align: center; }
.sub { text-align: center; color: #aaa; font-size: 18px; margin-bottom: 20px; }
.card { background: #1c1c1c; padding: 20px; border-radius: 20px; text-align: center;
        box-shadow: 0px 0px 15px rgba(255,255,255,0.05); margin-bottom: 15px; }
.metric { font-size: 28px; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown('<div class="big-title">😵‍💫 DELULU CHAT ANALYZER</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Your chats. Your reality. Your delusion. 💀🔥</div>', unsafe_allow_html=True)

uploaded = st.file_uploader("📂 Drop your WhatsApp chat (.txt) file here", type=["txt"])


# ------------------ UNIVERSAL WHATSAPP PARSER ------------------
def parse_chat(file):
    data = file.read().decode("utf-8", errors="ignore")
    lines = data.split("\n")

    messages = []

    patterns = [
        r'^(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2})\s?-?\s(.*?):\s(.*)',
        r'^(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}:\d{2})\s?-?\s(.*?):\s(.*)',
        r'^\[(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}:\d{2})\]\s(.*?):\s(.*)',
        r'^(\d{1,2}/\d{1,2}/\d{2,4})\s(\d{1,2}:\d{2})\s-\s(.*?):\s(.*)',
    ]

    for line in lines:
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                date, time, user, msg = match.groups()
                try:
                    dt = datetime.strptime(date + " " + time, "%d/%m/%Y %H:%M")
                except:
                    try:
                        dt = datetime.strptime(date + " " + time, "%d/%m/%Y %H:%M:%S")
                    except:
                        continue
                messages.append([dt, user, msg])
                break

    df = pd.DataFrame(messages, columns=["Datetime", "User", "Message"])
    return df


if uploaded:
    df = parse_chat(uploaded)

    if df.empty:
        st.error("Bro this file format is not supported 😭 (WhatsApp format issue)")
        st.stop()

    # ------------------ BASIC PREP ------------------
    df["Date"] = df["Datetime"].dt.date
    df["Hour"] = df["Datetime"].dt.hour

    total_messages = len(df)
    total_words = df["Message"].apply(lambda x: len(x.split())).sum()
    total_emojis = df["Message"].apply(lambda x: len(re.findall(r'[😂😭🔥💀😍🥺🤣]', x))).sum()
    total_time_days = (df["Datetime"].max() - df["Datetime"].min()).days
    avg_per_day = total_messages / df["Date"].nunique()

    most_active_day = df["Date"].value_counts().idxmax()
    most_active_hour = df["Hour"].value_counts().idxmax()

    longest_msg = df.loc[df["Message"].str.len().idxmax()]["Message"]
    shortest_msg = df.loc[df["Message"].str.len().idxmin()]["Message"]

    # ------------------ 25 FEATURES CALC ------------------

    # 10 Emoji usage
    emojis = re.findall(r'[😂😭🔥💀😍🥺🤣]', " ".join(df["Message"]))
    emoji_counts = Counter(emojis)

    # 11 Top 10 words
    words = re.findall(r'\w+', " ".join(df["Message"]).lower())
    top_words = Counter(words).most_common(10)

    # 12 Mood score
    sentiments = df["Message"].apply(lambda x: TextBlob(x).sentiment.polarity)
    mood_score = sentiments.mean()

    # 13 Delulu score
    delulu_score = min(100, int((total_messages / 50) + (total_emojis * 2)))

    # 14 Ghosting index
    df_sorted = df.sort_values("Datetime")
    gaps = df_sorted["Datetime"].diff().dt.total_seconds().fillna(0)
    ghosting = int((gaps[gaps > 3600].count() / len(gaps)) * 100)

    # 15 Reply speed
    avg_reply_time = int(gaps[gaps > 0].mean() / 60) if len(gaps[gaps > 0]) > 0 else 0

    # 16 Late night %
    late_night = df[df["Hour"].between(0,5)]
    late_night_pct = int((len(late_night) / total_messages) * 100)

    # 17 Message streak
    streak = df["Date"].value_counts().max()

    # 18 One sided %
    user_counts = df["User"].value_counts()
    one_sided = int((user_counts.max() / total_messages) * 100)

    # 19 Attachments
    attachments = df["Message"].str.contains("attached|image|video", case=False).sum()

    # 20 Question rate
    questions = df["Message"].str.contains("\?").sum()

    # 21 LOL rate
    lol_rate = df["Message"].str.contains("lol|haha|😂", case=False).sum()

    # 22 Typing energy
    typing_energy = int(total_words / total_messages)

    # 23 Dryness
    dryness = int((df["Message"].apply(len).mean() / 100) * 100)

    # 24 Overthinking index
    overthink = df["Message"].str.contains("why|what if|overthink|\?", case=False).sum()

    # 25 Relationship vibe
    if delulu_score > 80 and mood_score > 0:
        vibe = "💘 In Love"
    elif one_sided > 70:
        vibe = "💔 One Sided"
    else:
        vibe = "😬 Situationship"

    # ------------------ UI CARDS ------------------
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(f'<div class="card">📩<div class="metric">{total_messages}</div>Total Messages</div>', unsafe_allow_html=True)
    with col2: st.markdown(f'<div class="card">📝<div class="metric">{total_words}</div>Total Words</div>', unsafe_allow_html=True)
    with col3: st.markdown(f'<div class="card">😵‍💫<div class="metric">{total_emojis}</div>Total Emojis</div>', unsafe_allow_html=True)
    with col4: st.markdown(f'<div class="card">⏳<div class="metric">{total_time_days} days</div>Chat Time</div>', unsafe_allow_html=True)

    col5, col6, col7, col8 = st.columns(4)
    with col5: st.markdown(f'<div class="card">📆<div class="metric">{avg_per_day:.1f}</div>Avg Msg/Day</div>', unsafe_allow_html=True)
    with col6: st.markdown(f'<div class="card">🗓<div class="metric">{most_active_day}</div>Most Active Day</div>', unsafe_allow_html=True)
    with col7: st.markdown(f'<div class="card">🕒<div class="metric">{most_active_hour}:00</div>Most Active Hour</div>', unsafe_allow_html=True)
    with col8: st.markdown(f'<div class="card">📜<div class="metric">{len(longest_msg)}</div>Longest Msg Length</div>', unsafe_allow_html=True)

    # ------------------ METERS ------------------
    st.subheader("😵‍💫 Delulu Level")
    st.progress(delulu_score)
    st.write(f"Delulu Score: {delulu_score}/100")

    st.subheader("👻 Ghosting Index")
    st.progress(ghosting)
    st.write(f"Ghosting Level: {ghosting}%")

    st.subheader("🧊 Dryness Level")
    st.progress(min(100, dryness))
    st.write(f"Dryness: {min(100, dryness)}%")

    st.subheader("🌙 Late Night Chat %")
    st.write(f"{late_night_pct}% messages between 12AM – 5AM")

    st.subheader("⚡ Avg Reply Speed")
    st.write(f"{avg_reply_time} minutes")

    st.subheader("🔥 Message Streak")
    st.write(f"Max messages in a day: {streak}")

    st.subheader("😬 One Sided Chat %")
    st.write(f"{one_sided}% from one person")

    st.subheader("📎 Attachments")
    st.write(f"{attachments} attachments shared")

    st.subheader("❓ Question Rate")
    st.write(f"{questions} questions asked")

    st.subheader("😂 LOL Rate")
    st.write(f"{lol_rate} laughs")

    st.subheader("⚡ Typing Energy")
    st.write(f"{typing_energy} words/message")

    st.subheader("🧠 Overthinking Index")
    st.write(f"{overthink} overthinking messages")

    # ------------------ EMOJI CHART ------------------
    st.subheader("😂 Emoji Usage")
    if emoji_counts:
        fig, ax = plt.subplots()
        ax.bar(emoji_counts.keys(), emoji_counts.values())
        st.pyplot(fig)
    else:
        st.write("No emojis found 😭")

    # ------------------ TOP WORDS ------------------
    st.subheader("🔝 Top 10 Words")
    st.write(pd.DataFrame(top_words, columns=["Word", "Count"]))

    # ------------------ FINAL VIBES ------------------
    st.subheader("💘 Relationship Vibe")
    st.success(vibe)

    st.subheader("🧠 Final Verdict")
    if delulu_score > 80:
        st.error("BRO YOU ARE CERTIFIED DELULU 💀🔥")
    elif delulu_score > 50:
        st.warning("High delulu energy 😭")
    else:
        st.success("Still in reality… for now 😏")
