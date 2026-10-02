"""Combine face and speech emotion into one steady traffic-light reading.

Face scores arrive many times a second and are noisy, so they are smoothed.
Speech arrives once per utterance and fades out over `speech_ttl` seconds.
The light only changes after the new colour has held for `hold_seconds`, so
the robot doesn't twitch between poses.
"""

import time

from reachy_emotion import EMOTIONS, EMOTION_TO_LIGHT


class EmotionFusion:
    def __init__(self, face_smoothing=0.15, speech_weight=0.4, speech_ttl=8.0,
                 confidence_threshold=0.45, hold_seconds=1.0, face_timeout=1.5):
        self.face_smoothing = face_smoothing
        self.speech_weight = speech_weight
        self.speech_ttl = speech_ttl
        self.confidence_threshold = confidence_threshold
        self.hold_seconds = hold_seconds
        self.face_timeout = face_timeout

        self.face_scores = None
        self.face_time = 0.0
        self.speech = None          # latest EmotionPredictor result
        self.speech_time = 0.0

        self.light = None           # the steady, shown light
        self._candidate = None
        self._candidate_since = 0.0

    def update_face(self, scores):
        if self.face_scores is None:
            self.face_scores = dict(scores)
        else:
            a = self.face_smoothing
            self.face_scores = {e: (1 - a) * self.face_scores[e] + a * scores[e] for e in EMOTIONS}
        self.face_time = time.time()

    def update_speech(self, result):
        self.speech = result
        self.speech_time = time.time()

    def current(self):
        """Return the fused reading, or None if there's nothing to go on"""
        now = time.time()
        face_ok = self.face_scores is not None and now - self.face_time < self.face_timeout
        speech_age = now - self.speech_time
        speech_ok = self.speech is not None and speech_age < self.speech_ttl

        if not face_ok and not speech_ok:
            self.face_scores = None
            self._settle(None, now)
            return None

        if speech_ok:
            speech_scores = {d["label"]: d["score"] for d in self.speech["all_emotions"]}

        if face_ok and speech_ok:
            w = self.speech_weight * (1 - speech_age / self.speech_ttl)
            scores = {e: (1 - w) * self.face_scores[e] + w * speech_scores.get(e, 0.0) for e in EMOTIONS}
        elif face_ok:
            scores = dict(self.face_scores)
        else:
            scores = {e: speech_scores.get(e, 0.0) for e in EMOTIONS}

        emotion = max(scores, key=scores.get)
        confidence = scores[emotion]

        # face and words confidently disagree (e.g. smiling while saying something
        # hostile) — often sarcasm or masking, which is exactly what's hard to read
        mixed = face_ok and speech_ok and self._lights_disagree(speech_scores)

        if mixed or confidence < self.confidence_threshold:
            raw_light = "AMBER"
        else:
            raw_light = EMOTION_TO_LIGHT[emotion]
        self._settle(raw_light, now)

        return {
            "emotion": emotion,
            "confidence": confidence,
            "light": self.light or raw_light,
            "uncertain": confidence < self.confidence_threshold,
            "mixed": mixed,
            "scores": scores,
            "sources": [s for s, ok in (("face", face_ok), ("speech", speech_ok)) if ok],
            "transcript": self.speech["text"] if speech_ok else None,
        }

    def _lights_disagree(self, speech_scores):
        face_top = max(self.face_scores, key=self.face_scores.get)
        speech_top = max(speech_scores, key=speech_scores.get)
        return (self.face_scores[face_top] >= self.confidence_threshold
                and speech_scores[speech_top] >= self.confidence_threshold
                and EMOTION_TO_LIGHT[face_top] != EMOTION_TO_LIGHT[speech_top])

    def _settle(self, raw_light, now):
        if raw_light == self.light:
            self._candidate = None
            return
        if raw_light != self._candidate:
            self._candidate, self._candidate_since = raw_light, now
        elif now - self._candidate_since >= self.hold_seconds:
            self.light, self._candidate = raw_light, None
