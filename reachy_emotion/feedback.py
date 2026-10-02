"""Turn a reading into one short, gentle sentence for the child.

Uses the local Ollama model (same one as Sentiment-Guide-Chrome-Ext) when it's
running; otherwise falls back to fixed hints. Wording is always hedged —
expression models are often wrong, and the child should know that.
"""

import json
import queue
import threading

import requests

OLLAMA_URL = "http://localhost:11434"
KEEP_ALIVE = "30m"   # the first load can take minutes on a laptop; keep it in memory

FALLBACK = {
    "joy": "They look happy. It's probably a good time to chat.",
    "neutral": "They seem calm. You could say hi or ask how they are.",
    "surprise": "They might be surprised. Give them a moment.",
    "sadness": "They might be feeling down. You could ask if they're okay.",
    "fear": "They might be worried. Speak gently and give them space.",
    "anger": "They might be upset. It could be a good time to step back.",
    "disgust": "They might not like something. Maybe check in or change topic.",
}
UNSURE = "I'm not sure how they're feeling. Watch and listen a bit more."
MIXED = "Their face and words don't match. They might be joking, or hiding how they feel."


def ollama_model():
    """Return an installed gemma model name, or None if Ollama isn't reachable"""
    try:
        names = [m["name"] for m in requests.get(f"{OLLAMA_URL}/api/tags", timeout=1).json().get("models", [])]
    except (requests.RequestException, ValueError):
        return None
    return next((n for n in names if n.startswith("gemma")), names[0] if names else None)


class FeedbackGenerator:
    def __init__(self, use_llm=True, speak=False):
        self.model = ollama_model() if use_llm else None
        self.text = ""
        self._last_key = None
        self._requests = queue.Queue()
        self._speech = queue.Queue() if speak else None
        threading.Thread(target=self._llm_worker, daemon=True).start()
        if speak:
            threading.Thread(target=self._speak_worker, daemon=True).start()

    def update(self, reading):
        """Refresh the hint when the steady light, emotion or transcript changes"""
        if reading is None:
            key = None
        else:
            key = (reading["light"], reading["emotion"], reading["uncertain"], reading["mixed"], reading["transcript"])
        if key == self._last_key:
            return
        self._last_key = key

        if reading is None:
            self.text = ""
            return
        if reading["mixed"]:
            self._set(MIXED)
        elif reading["uncertain"]:
            self._set(UNSURE)
        else:
            self._set(FALLBACK[reading["emotion"]])
        if self.model:
            self._requests.put(reading)

    def _set(self, text):
        if text == self.text:
            return
        self.text = text
        if self._speech is not None:
            self._speech.put(text)

    def _llm_worker(self):
        if not self.model:
            return
        # load the model up front; fixed hints are shown until this finishes
        try:
            requests.post(f"{OLLAMA_URL}/api/generate", timeout=600,
                          json={"model": self.model, "prompt": "", "keep_alive": KEEP_ALIVE})
            print(f"Feedback: {self.model} loaded")
        except requests.RequestException as err:
            print("Feedback: could not load Ollama model, using fixed hints —", err)
            self.model = None
            return

        while True:
            reading = self._requests.get()
            while not self._requests.empty():   # only answer the newest reading
                reading = self._requests.get_nowait()
            prompt = (
                "You help an autistic child read another person's feelings during a conversation. "
                "A camera and microphone produced this reading, which may be wrong:\n"
                f"- Most likely emotion: {reading['emotion']} ({reading['confidence']:.0%} sure)\n"
                f"- Signal: {reading['light']} (GREEN safe, AMBER careful, RED step back)\n"
                f"- What they just said: {json.dumps(reading['transcript']) if reading['transcript'] else 'nothing heard'}\n"
                + ("- Their face and their words disagree — they may be joking, sarcastic or masking.\n"
                   if reading["mixed"] else "") + "\n"
                "Reply with ONE short sentence (max 20 words) for the child: what the person might be "
                "feeling and one simple thing to do. Use 'might' or 'seems' — never state it as fact. "
                "Warm, plain words. No emojis, no quotes."
            )
            try:
                res = requests.post(f"{OLLAMA_URL}/api/generate", timeout=90,
                                    json={"model": self.model, "prompt": prompt, "stream": False,
                                          "keep_alive": KEEP_ALIVE})
                text = res.json().get("response", "").strip().strip('"')
                if text and self._requests.empty():
                    self._set(text.splitlines()[0][:160])
            except (requests.RequestException, ValueError) as err:
                print("Feedback: Ollama request failed —", err)

    def _speak_worker(self):
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        while True:
            text = self._speech.get()
            while not self._speech.empty():
                text = self._speech.get_nowait()
            engine.say(text)
            engine.runAndWait()
