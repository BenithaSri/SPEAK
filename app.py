
import gradio as gr
from transformers import pipeline
import whisper
from collections import Counter
import matplotlib.pyplot as plt

# Load models
emotion_classifier = pipeline("audio-classification", model="superb/hubert-large-superb-er")
whisper_model = whisper.load_model("base")

def create_emotion_chart(labels, scores):
    emoji_map = {
        "hap": "😊 Happy", "sad": "😔 Sad", "neu": "😐 Neutral",
        "ang": "😠 Angry", "fea": "😨 Fear", "dis": "🤢 Disgust", "sur": "😮 Surprise"
    }
    color_map = {
        "hap": "#facc15", "sad": "#60a5fa", "neu": "#a1a1aa",
        "ang": "#ef4444", "fea": "#818cf8", "dis": "#14b8a6", "sur": "#f472b6"
    }
    display_labels = [emoji_map.get(label, label) for label in labels]
    colors = [color_map.get(label, "#60a5fa") for label in labels]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.barh(display_labels, scores, color=colors, edgecolor="black", height=0.5)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2, f"{score:.2f}", va='center', fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_title("🎭 Emotion Confidence Scores", fontsize=13, pad=10)
    ax.invert_yaxis()
    ax.set_facecolor("#f9fafb")
    fig.patch.set_facecolor("#f9fafb")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='x', colors='gray')
    ax.tick_params(axis='y', colors='gray')
    return fig

def generate_next_moves(dominant_emotion, conf_score, transcript=""):
    suggestions = []
    harsh_words = ["bad", "ugly", "terrible", "hate", "worst"]
    positive_tone_negative_words = any(word in transcript.lower() for word in harsh_words) if "happiness" in dominant_emotion else False
    if 'sadness' in dominant_emotion:
        suggestions.append("Your tone feels low — try lifting the pitch slightly to bring more warmth.")
        suggestions.append("Even if the words are positive, a brighter tone helps convey enthusiasm.")
    elif 'happiness' in dominant_emotion and conf_score >= 80:
        suggestions.append("Nice energy! Try modulating your tone even more for emphasis in key moments.")
        suggestions.append("Experiment with subtle emotional shifts as you speak for more depth.")
    elif 'neutral' in dominant_emotion:
        suggestions.append("Add inflection to break a monotone pattern — especially at the ends of sentences.")
        suggestions.append("Highlight your message by stressing emotionally important words.")
    elif conf_score < 50:
        suggestions.append("Try exaggerating vocal ups and downs when reading to unlock more expression.")
        suggestions.append("Slow down slightly and stretch certain words to vary your delivery.")
    else:
        suggestions.append("Keep practicing tone variation — you’re building a solid base.")
    if positive_tone_negative_words:
        suggestions.append("Your tone was upbeat, but the word choices were harsh — aim to align both for better impact.")
    return "\n- " + "\n- ".join(suggestions)

def generate_personacoach_report(emotions, transcript):
    report = "## 📝 **Your PersonaCoach Report**\n---\n\n"
    report += "### 🗒️ **What You Said:**\n"
    report += f"> _{transcript.strip()}_\n\n"
    label_map = {
        'hap': '😊 happiness', 'sad': '😔 sadness', 'neu': '😐 neutral',
        'ang': '😠 anger', 'fea': '😨 fear', 'dis': '🤢 disgust', 'sur': '😮 surprise'
    }
    for e in emotions:
        e['emotion'] = label_map.get(e['label'], e['label'])
    scores = [s['score'] for s in emotions]
    top_score = max(scores)
    conf_score = int(top_score * 100)
    meaningful_emotions = [(e['emotion'], e['score']) for e in emotions if e['score'] >= 0.2]
    emotion_labels = [e[0] for e in meaningful_emotions]
    dominant_emotion = emotion_labels[0] if emotion_labels else "neutral"

    report += f"### 🎯 **Tone Strength:**\n- Your tone scored **{conf_score}/100** in clarity.\n\n"
    report += "### 🗣️ **Emotion & Delivery:**\n"
    if meaningful_emotions:
        emotions_str = ", ".join([f"**{label}** ({score:.2f})" for label, score in meaningful_emotions])
        report += f"- Emotionally, your voice showed: {emotions_str}\n"
    else:
        report += "- Your tone wasn’t clearly expressive. Try reading with a bit more emphasis or emotion.\n"
    filler_words = ["um", "uh", "like", "you know", "so", "actually", "basically", "literally"]
    words = transcript.lower().split()
    total_words = len(words)
    filler_count = sum(words.count(fw) for fw in filler_words)
    filler_ratio = filler_count / total_words if total_words > 0 else 0

    report += "\n### 💬 **Pausing Style (e.g., 'um', 'like', 'you know'):**\n"
    report += f"- You used **{filler_count}** hesitation phrases out of **{total_words}** words.\n"
    if filler_ratio > 0.06:
        report += "- Try pausing instead of using fillers — it builds stronger presence.\n"
    elif filler_ratio > 0.03:
        report += "- A few slipped in. Practice holding space with silence instead.\n"
    else:
        report += "- Great fluency — you stayed focused and controlled.\n"

    report += "\n### ✅ **What You're Doing Well:**\n"
    if top_score >= 0.75 and filler_ratio < 0.03:
        report += "- Confident tone and smooth delivery — keep it up!\n"
    else:
        report += "- You’re on track. Keep refining tone and pacing.\n"

    report += "\n### 🧭 **Next Moves:**\n"
    report += generate_next_moves(dominant_emotion, conf_score, transcript) + "\n"
    return report

def analyze_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    transcript = result['text']
    emotion_results = emotion_classifier(audio_path)
    labels = [r['label'] for r in emotion_results]
    scores = [r['score'] for r in emotion_results]
    fig = create_emotion_chart(labels, scores)
    report = generate_personacoach_report(emotion_results, transcript)
    return transcript, fig, report

with gr.Blocks(title="SPEAK: PersonaCoach", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    <div style="text-align:center; margin-bottom: 1rem;">
        <h1 style="font-size: 2.2rem; margin-bottom: 0.2rem;">🎤 SPEAK: PersonaCoach</h1>
        <p style="color: gray;">Your smart voice reflection tool — assess tone, confidence, and delivery</p>
    </div>
    """, elem_id="header")

    with gr.Row():
        with gr.Column(scale=4):
            audio_input = gr.Audio(type="filepath", label="🎧 Upload Your Voice (.wav)", elem_id="upload-audio")
        with gr.Column(scale=1, min_width=120):
            analyze_btn = gr.Button("🔍 Analyze", size="sm", elem_id="analyze-btn")

    gr.Markdown("## 🧠 Results", elem_id="results-header")

    with gr.Row(equal_height=True):
        with gr.Column(scale=2):
            feedback_output = gr.Markdown(label="📄 PersonaCoach Feedback", elem_id="report-section")
        with gr.Column(scale=1):
            emotion_plot = gr.Plot(label="🎭 Emotion Chart", elem_id="chart")

    analyze_btn.click(
        fn=analyze_audio,
        inputs=audio_input,
        outputs=[gr.Textbox(visible=False), emotion_plot, feedback_output]
    )

app.launch()
