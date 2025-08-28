from transformers import pipeline
import math


emotion_classifier = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", return_all_scores=True)


def emotion_classify(input):
    """
    anger ; disgut ; fear ; joy ; neutral ; sadness ; surprise

    """
    return emotion_classifier(input)[0]

def index_emotionnal_charge(d):
    emotion_weights = {
    "admiration": 0.3,
    "amusement": 0.35,
    "anger": -1,
    "annoyance": -0.4,
    "approval": 0.4,
    "caring": 0.45,
    "confusion": -0.2,
    "curiosity": 0.3,
    "desire": 0.9,
    "disappointment": -0.5,
    "disapproval": -0.6,
    "disgust": -0.65,
    "embarrassment": -0.3,
    "excitement": 0.6,
    "fear": -0.6,
    "gratitude": 0.5,
    "grief": -0.8,
    "joy": 1,
    "love": 1.0,
    "nervousness": -0.4,
    "optimism": 0.8,
    "pride": 0.4,
    "realization": 0.4,
    "relief": 0.5,
    "remorse": -0.5,
    "sadness": -0.8,
    "surprise": 0.2,  # surprise peut être positive ou négative, ici neutre-positive
    "neutral": 0.0
    }


    def compute_emotion_score(emotions, weights):
        return sum(e["score"] * weights.get(e["label"], 0.0) for e in emotions)

    def apply_polarity_boost(score):
        return math.tanh(score * 3)

    raw_score = compute_emotion_score(d, emotion_weights)
    boosted_score = apply_polarity_boost(raw_score)
    return boosted_score

print(emotion_classify("J'aime les samedi pluvieux, ca rajoute une touche de... poésie"))