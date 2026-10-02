"""Background listener: waits for speech, transcribes it, scores its emotion.

Reuses EmotionPredictor from Sentiment-Analysis-Detection-v1 so both projects
share one text-emotion model. Audio is kept in memory only, never written out.
"""

import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import whisper

_SIBLING = Path(__file__).resolve().parents[2] / "Sentiment-Analysis-Detection-v1"
sys.path.insert(0, str(_SIBLING))
from src.inference import EmotionPredictor  # noqa: E402

SAMPLE_RATE = 16000
BLOCK = 1024


class SpeechListener(threading.Thread):
    def __init__(self, on_result, on_status=None, whisper_model_size="base",
                 silence_seconds=1.2, min_speech_seconds=0.5, device=None, robot_media=None):
        """device: sounddevice input name/index (None = system default).
        robot_media: a Reachy Mini's `media` to listen through its mics
        instead of the laptop's"""
        super().__init__(daemon=True)
        self.device = device
        self.robot_media = robot_media
        self.on_result = on_result
        self.on_status = on_status or (lambda s: None)
        self.silence_blocks = int(silence_seconds * SAMPLE_RATE / BLOCK)
        self.min_samples = int(min_speech_seconds * SAMPLE_RATE)
        self._stopping = threading.Event()

        print(f"Loading Whisper ({whisper_model_size})...")
        self.whisper_model = whisper.load_model(whisper_model_size)
        self.predictor = EmotionPredictor()

    def stop(self):
        self._stopping.set()

    def run(self):
        blocks = queue.Queue()

        def callback(indata, frames, time_info, status):
            blocks.put(indata[:, 0].copy())

        if self.robot_media is not None:
            from reachy_emotion.robot import RobotMic
            stream = RobotMic(self.robot_media, SAMPLE_RATE, BLOCK).open(blocks)
        else:
            stream = sd.InputStream(device=self.device, samplerate=SAMPLE_RATE, channels=1, dtype=np.float32,
                                    blocksize=BLOCK, callback=callback)
        with stream:
            threshold = self._calibrate(blocks)
            self.on_status("listening")

            speech, silent_run = [], 0
            while not self._stopping.is_set():
                try:
                    block = blocks.get(timeout=0.2)
                except queue.Empty:
                    continue
                loud = np.sqrt(np.mean(block ** 2)) > threshold

                if not speech:
                    if loud:
                        speech, silent_run = [block], 0
                        self.on_status("hearing speech")
                    continue

                speech.append(block)
                silent_run = 0 if loud else silent_run + 1
                if silent_run >= self.silence_blocks:
                    self._process(np.concatenate(speech))
                    speech = []
                    # drop audio that queued up while Whisper was busy
                    while not blocks.empty():
                        blocks.get_nowait()
                    self.on_status("listening")

    def _calibrate(self, blocks, seconds=1.0):
        """Set the speech threshold from a second of room noise"""
        self.on_status("calibrating mic (stay quiet)")
        levels, end = [], time.time() + seconds
        while time.time() < end:
            try:
                levels.append(np.sqrt(np.mean(blocks.get(timeout=0.5) ** 2)))
            except queue.Empty:
                pass
        noise = float(np.median(levels)) if levels else 0.0
        return max(0.01, noise * 3)

    def _process(self, audio):
        if len(audio) < self.min_samples:
            return
        self.on_status("transcribing")
        text = self.whisper_model.transcribe(audio, language="en", fp16=False)["text"].strip()
        if not text:
            return
        self.on_result(self.predictor.predict(text))
