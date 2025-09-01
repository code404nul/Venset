import json
import random
import re
import os
import sys
from difflib import SequenceMatcher
from ollama import chat, ChatResponse

CATEGORIES = [
    "food and drinks", "childhood memories", "daily routines",
    "friendship", "pop culture", "funny moments",
    "school and studies", "hobbies", "annoyances",
    "music", "sports", "technology", "shopping",
    "pets", "weather", "transportation", "weekend plans",
    "gadgets", "money", "clothes", "health",
    "movies", "series", "gaming", "holidays",
    "morning habits", "bedtime habits", "neighbors", "family traditions"
]

REMOVE_CHARS = ['"', "*", "“", "”"]
DEFAULT_TOTAL_CONVERSATIONS = 400
TURNS_PER_CONVERSATION = 6
OUTPUT_FILE = "conversations.json"

def talk_gemma(prompt: str) -> str:
    """Send a prompt to Gemma model and clean output."""
    response: ChatResponse = chat(model='gemma3:27b-it-qat', messages=[
        { 'role': 'user', 'content': prompt.strip() }
    ])
    output = response['message']['content']
    for char in REMOVE_CHARS:
        output = output.replace(char, "")
    return output.strip()

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    return text.strip()

def too_similar(a: str, b: str, threshold: float = 0.8) -> bool:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio() > threshold

SEEN_THEMES = set()

def generate_subject():
    """Generate a unique theme for a conversation."""
    for _ in range(5):  # retry few times if too similar
        category = random.choice(CATEGORIES)
        prompt = (
            f"Give me one short, casual conversation theme two friends could talk about.\n"
            f"Category: {category}.\n"
            "Keep it simple, ordinary, and human. Do not explain, just output the theme."
        )
        theme = talk_gemma(prompt)
        if theme and not any(too_similar(theme, s) for s in SEEN_THEMES):
            SEEN_THEMES.add(theme)
            return {"category": category, "theme": theme}
    return None

def human_talk(theme: str, prev: str | None = None) -> str:
    prompt = (
        f"You are with your friend. You are a bit tired, not in the mood for deep talks, "
        f"but you want to keep the conversation going. You are talking about: {theme}. "
        f"Keep it short, natural, and casual."
    )
    if prev:
        prompt += f"\nYour friend said: '{prev}'"
    return talk_gemma(prompt)

def ai_talk(theme: str, prev: str | None = None) -> str:
    prompt = (
        f"Friend mentioned: {theme}. Respond in a short, warm, and alive way. "
        f"Your voice should have character, maybe poetic, tender, or witty. "
        f"Keep it casual and brief."
    )
    if prev:
        prompt += f"\nYour friend said: '{prev}'"
    return talk_gemma(prompt)

def build_conversation(theme: str, n_turns: int = TURNS_PER_CONVERSATION) -> list:
    """Build a back-and-forth conversation between two friends."""
    convo = []
    last_msg = None
    for i in range(n_turns):
        if i % 2 == 0:
            msg = human_talk(theme, last_msg)
            convo.append({"role": "friend1", "text": msg})
        else:
            msg = ai_talk(theme, last_msg)
            convo.append({"role": "friend2", "text": msg})
        last_msg = msg
    return convo

def save_progress(conversations: list):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)

def main():
    
    conversations = []
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                conversations = json.load(f)
            print(f"🔄 Resuming from {len(conversations)} saved conversations...")
            for c in conversations:
                SEEN_THEMES.add(c["theme"])
        except Exception:
            print("⚠️ Could not load existing progress, starting fresh.")

    total = DEFAULT_TOTAL_CONVERSATIONS
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        total = int(sys.argv[1])

    while len(conversations) < total:
        subject = generate_subject()
        if subject:
            convo = build_conversation(subject["theme"], n_turns=TURNS_PER_CONVERSATION)
            conversations.append({
                "category": subject["category"],
                "theme": subject["theme"],
                "conversation": convo
            })
            print(f"[{len(conversations)}/{total}] {subject['theme']}")

            save_progress(conversations)

    print(f"✅ {total} conversations saved in {OUTPUT_FILE}")

    print("\n--- Preview of first 3 conversations ---")
    for conv in conversations[:3]:
        print(f"\nTheme: {conv['theme']} ({conv['category']})")
        for turn in conv['conversation']:
            print(f"{turn['role']}: {turn['text']}")

if __name__ == "__main__":
    main()