from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import re
import numpy as np
from collections import Counter
from textblob import TextBlob
from datetime import timedelta
import string
import uuid

app = FastAPI(title="Delulu Meter Ultra Engine 😈")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULT_STORE = {}

# ------------------ HELPERS ------------------

def chunkify(lst, size=1000):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

def parse_chunk(lines):
    data = []
    pattern = r'^(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}\s?(?:am|pm|AM|PM)?)\s-\s(.*?):\s(.*)$'
    for line in lines:
        match = re.match(pattern, line.strip())
        if match:
            data.append(match.groups())
    if not data:
        return pd.DataFrame(columns=["Date","Time","Author","Message"])
    df = pd.DataFrame(data, columns=["Date","Time","Author","Message"])
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce', dayfirst=True)
    df = df.dropna(subset=['DateTime'])
    df['Message'] = df['Message'].astype(str)
    return df

def get_sentiment(text):
    try:
        return TextBlob(text).sentiment.polarity
    except:
        return 0.0

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

# ------------------ CORE ANALYSIS ------------------

def process_full_chat(file_bytes, task_id):
    try:
        text = file_bytes.decode("utf-8", errors="ignore")
        lines = text.split("\n")

        all_dfs = []
        for chunk in chunkify(lines, 1000):
            df_chunk = parse_chunk(chunk)
            if not df_chunk.empty:
                all_dfs.append(df_chunk)

        if not all_dfs:
            RESULT_STORE[task_id] = {"status": "error", "message": "No valid chat data found."}
            return

        df = pd.concat(all_dfs)

        # -------- FEATURES --------
        df['Sentiment'] = df['Message'].apply(get_sentiment)

        initiation = df.groupby(df['DateTime'].dt.date).first()['Author'].value_counts()
        initiation_index = (initiation / initiation.sum() * 100).round(1).to_dict()

        reply_times = {}
        df_sorted = df.sort_values('DateTime')
        for i in range(1, len(df_sorted)):
            if df_sorted.iloc[i]['Author'] != df_sorted.iloc[i-1]['Author']:
                diff = (df_sorted.iloc[i]['DateTime'] - df_sorted.iloc[i-1]['DateTime']).total_seconds() / 60
                reply_times.setdefault(df_sorted.iloc[i]['Author'], []).append(diff)
        reply_times = {k: round(np.mean(v),1) for k,v in reply_times.items()}

        counts = df['Author'].value_counts()
        ratio = counts.min() / counts.max() if len(counts) > 1 else 1
        engagement_balance = "Balanced participation ✨" if ratio > 0.8 else "One-sided energy 💀"

        sentiment_timeline = df.groupby(df['DateTime'].dt.date)['Sentiment'].mean().round(3).to_dict()
        sentiment_variability = round(df['Sentiment'].std(),3)

        df['Hour'] = df['DateTime'].dt.hour
        late_activity = df[df['Hour'].between(0,4)]['Author'].value_counts().to_dict()

        emojis = emoji_usage(df)

        communication_style = {}
        for auth in df['Author'].unique():
            msgs = df[df['Author']==auth]['Message']
            avg_len = np.mean([len(m.split()) for m in msgs])
            q_marks = sum(m.count('?') for m in msgs)
            style = "Concise 📝" if avg_len < 5 else "Elaborative 📚" if avg_len > 20 else "Balanced ⚖️"
            if q_marks > len(msgs)*0.3:
                style += " + Curious 🤔"
            communication_style[auth] = style

        gaps = {}
        for i in range(1, len(df_sorted)):
            gap = (df_sorted.iloc[i]['DateTime'] - df_sorted.iloc[i-1]['DateTime']).total_seconds() / 3600
            gaps.setdefault(df_sorted.iloc[i]['Author'], []).append(gap)
        interaction_gaps = {k: round(max(v),2) for k,v in gaps.items()}

        text_all = " ".join(df['Message']).lower().translate(str.maketrans('', '', string.punctuation))
        words = [w for w in text_all.split() if len(w)>3]
        keyword_themes = Counter(words).most_common(15)

        harmony = round((( (1 if ratio > 0.8 else 0.5) + (1 if np.mean(list(reply_times.values())) < 30 else 0.6) + ((df['Sentiment'].mean()+1)/2) ) / 3)*100,1)

        delulu_level = "Low 😌" if harmony > 75 else "Medium 😬" if harmony > 50 else "High 😭"

        RESULT_STORE[task_id] = {
            "status": "done",
            "total_messages": len(df),
            "initiation_index": initiation_index,
            "reply_times": reply_times,
            "engagement_balance": engagement_balance,
            "sentiment_timeline": sentiment_timeline,
            "sentiment_variability": sentiment_variability,
            "late_activity": late_activity,
            "emoji_usage": emojis,
            "communication_style": communication_style,
            "interaction_gaps": interaction_gaps,
            "keyword_themes": keyword_themes,
            "harmony_score": harmony,
            "delulu_level": delulu_level
        }

    except Exception as e:
        RESULT_STORE[task_id] = {"status": "error", "message": str(e)}

# ------------------ API ------------------

@app.get("/ping")
def ping():
    return {"status": "alive"}

@app.post("/analyze-chat")
async def analyze_chat(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    file_bytes = await file.read()
    task_id = str(uuid.uuid4())
    RESULT_STORE[task_id] = {"status": "processing"}
    background_tasks.add_task(process_full_chat, file_bytes, task_id)
    return {"status": "processing", "task_id": task_id}

@app.get("/result/{task_id}")
def get_result(task_id: str):
    return RESULT_STORE.get(task_id, {"status": "processing"})
