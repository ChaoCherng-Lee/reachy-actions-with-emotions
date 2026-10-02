"""Webcam + speech emotion feedback prototype, designed to move onto a Reachy Mini."""

# Shared by every module: the 7 labels the text model (j-hartmann DistilRoBERTa)
# uses. Face scores are mapped onto these so the two sources can be combined.
EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# Same mapping as Sentiment-Analysis-Detection-v1/src/inference.py
EMOTION_TO_LIGHT = {
    "anger": "RED",
    "disgust": "RED",
    "fear": "AMBER",
    "sadness": "AMBER",
    "surprise": "AMBER",
    "joy": "GREEN",
    "neutral": "GREEN",
}
