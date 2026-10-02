"""Find the most prominent face in a frame and score its expression."""

import cv2
import numpy as np
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

from reachy_emotion import EMOTIONS

# HSEmotion is trained on AffectNet's 8 classes; fold them into the text model's 7
_TO_COMMON = {
    "Anger": "anger",
    "Contempt": "disgust",
    "Disgust": "disgust",
    "Fear": "fear",
    "Happiness": "joy",
    "Neutral": "neutral",
    "Sadness": "sadness",
    "Surprise": "surprise",
}


class FaceEmotionDetector:
    def __init__(self, model_name="enet_b0_8_best_vgaf"):
        print(f"Loading face expression model: {model_name}...")
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.recognizer = HSEmotionRecognizer(model_name=model_name)
        self.labels = [_TO_COMMON[self.recognizer.idx_to_class[i]]
                       for i in range(len(self.recognizer.idx_to_class))]
        print("Face model loaded")

    def detect(self, frame_bgr):
        """Return ((x, y, w, h), {emotion: score}) for the largest face, or None"""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6, minSize=(80, 80)
        )
        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # pad the box — the model was trained on loosely cropped faces
        pad = int(0.15 * w)
        fh, fw = frame_bgr.shape[:2]
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(fw, x + w + pad), min(fh, y + h + pad)
        face_rgb = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)

        _, scores = self.recognizer.predict_emotions(face_rgb, logits=False)

        common = dict.fromkeys(EMOTIONS, 0.0)
        for label, score in zip(self.labels, np.asarray(scores).ravel()):
            common[label] += float(score)
        return (int(x), int(y), int(w), int(h)), common
