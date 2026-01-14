from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import re
import numpy as np
from collections import Counter
from textblob import TextBlob
from datetime import timedelta
import string

app = FastAPI(title="Delulu Meter API")

# -------- CORS --------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- CONFIG --------
MAX_LINES = 5000   # hard safety limit for Render free tier

# ------------------ PARSER ------------------
def parse_whatsapp_chat_bytes(file_bytes):
    text = file_bytes.decode("utf-8", errors="ignore")
    lines = text.split('\n')[:MAX_LINES]
    data = []

    pattern = r'^(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}\s?(?:am|pm|AM|PM)?)\s-\s(.*?):\s(.*)$'

    for line in lines:
        match = re.match(pattern, line.strip())
        if match:
            data.append(match.groups())

    df = pd.DataFrame(data, columns=["Date", "Time", "Author", "Message"])
    if df.empty:
        return df

    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce', dayfirst=True)
    df = df.dropna(subset=['DateTime'])
    df['Message'] = df['Message'].astype(str)

    return df

# ------------------ UTILITIES ------------------
def get_sentiment(text):
    try:
        return TextBlob(text).sentiment.polarity
    except:
        return 0.0

def initiation_index(df):
    df['Day'] = df['DateTime'].dt.date
    starters = df.groupby('Day').first()['Author'].value_counts()
    total = starters.sum()
    return (starters / total * 100).round(1).to_dict()

def compute_reply_times(df):
    df = df.sort_values('DateTime')
    reply = {}
    for i in range(1, len(df)):
        if df.iloc[i]['Author'] != df.iloc[i-1]['Author']:
            diff = (df.iloc[i]['DateTime'] - df.iloc[i-1]['DateTime']).total_seconds() / 60
            reply.setdefault(df.iloc[i]['Author'], []).append(diff)
    return {k: round(np.mean(v),1) for k,v in reply.items()}

def engagement_balance(df):
    counts = df['Author'].value_counts()
    if len(counts) < 2:
        return "Not enough data"
    ratio = counts.min() / counts.max()
    return "Balanced participation ✨" if ratio > 0.8 else "Participation levels vary 📊"

def sentiment_timeline(df):
    df['Sentiment'] = df['Message'].apply(get_sentiment)
    timeline = df.groupby(df['DateTime'].dt.date)['Sentiment'].mean()
    return {str(k): round(v,3) for k,v in timeline.items()}

def sentiment_variability(df):
    return round(df['Message'].apply(get_sentiment).std(), 3)

def continuity_index(df, gap_minutes=30):
    df = df.sort_values('DateTime')
    longest = timedelta(0)
    start = df.iloc[0]['DateTime']
    for i in range(1, len(df)):
        if df.iloc[i]['DateTime'] - df.iloc[i-1]['DateTime'] > timedelta(minutes=gap_minutes):
            longest = max(longest, df.iloc[i-1]['DateTime'] - start)
            start = df.iloc[i]['DateTime']
    longest = max(longest, df.iloc[-1]['DateTime'] - start)
    return str(longest)

def late_activity(df):
    df['Hour'] = df['DateTime'].dt.hour
    return df[df['Hour'].between(0,4)]['Author'].value_counts().to_dict()

def emoji_usage(df):
    emoji_pattern = re.compile("[" 
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
    counts = {}
    for _, row in df.iterrows():
        counts[row['Author']] = counts.get(row['Author'], 0) + len(emoji_pattern.findall(row['Message']))
    return counts

def communication_style(df):
    styles = {}
    for auth in df['Author'].unique():
        msgs = df[df['Author']==auth]['Message']
        avg_len = np.mean([len(m.split()) for m in msgs])
        q_marks = sum(m.count('?') for m in msgs)
        emojis = emoji_usage(df).get(auth,0)

        style = "Concise 📝" if avg_len < 5 else "Elaborative 📚" if avg_len > 20 else "Balanced ⚖️"
        if q_marks > len(msgs)*0.3:
            style += " + Inquisitive 🤔"
        if emojis > len(msgs)*0.5:
            style += " + Expressive 😄"
        styles[auth] = style
    return styles

def interaction_gaps(df):
    df = df.sort_values('DateTime')
    gaps = {}
    for i in range(1, len(df)):
        gap = (df.iloc[i]['DateTime'] - df.iloc[i-1]['DateTime']).total_seconds() / 3600
        gaps.setdefault(df.iloc[i]['Author'], []).append(gap)
    return {k: round(max(v),2) for k,v in gaps.items()}

def keyword_themes(df, top_n=10):
    text = " ".join(df['Message']).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = [w for w in text.split() if len(w)>3]
    return Counter(words).most_common(top_n)

def interaction_harmony_index(df):
    balance = 1 if "balanced" in engagement_balance(df).lower() else 0.5
    sentiment = df['Message'].apply(get_sentiment).mean()
    reply = compute_reply_times(df)
    reply_score = 1 if reply and np.mean(list(reply.values())) < 30 else 0.6 if reply else 0.5
    return round(((balance + reply_score + ((sentiment+1)/2)) / 3)*100,1)

# ------------------ HEALTH CHECK ------------------
@app.get("/ping")
def ping():
    return {"status": "alive"}

# ------------------ API ENDPOINT ------------------
@app.post("/analyze-chat")
async def analyze_chat(file: UploadFile = File(...)):
    file_bytes = await file.read()
    df = parse_whatsapp_chat_bytes(file_bytes)

    if df.empty:
        return {"error": "Could not parse chat. Please upload valid WhatsApp export."}

    result = {
        "total_messages": len(df),
        "initiation_index": initiation_index(df),
        "reply_times": compute_reply_times(df),
        "engagement_balance": engagement_balance(df),
        "sentiment_timeline": sentiment_timeline(df),
        "sentiment_variability": sentiment_variability(df),
        "continuity_index": continuity_index(df),
        "late_activity": late_activity(df),
        "emoji_usage": emoji_usage(df),
        "communication_style": communication_style(df),
        "interaction_gaps": interaction_gaps(df),
        "keyword_themes": keyword_themes(df),
        "harmony_score": interaction_harmony_index(df)
    }

    return result
