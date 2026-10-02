"""Reachy emotion prototype — laptop camera + mic now, Reachy Mini later.

    python main.py                  # webcam + mic, on-screen virtual Reachy
    python main.py --no-mic         # face only (faster to start)
    python main.py --robot reachy   # also drive a Reachy Mini / MuJoCo sim
    python main.py --robot reachy --robot-media   # see and hear through the robot
    python main.py --speak          # read feedback aloud

Press q or Esc in the window to quit. Nothing is saved to disk.
"""

import argparse
import sys
import time

import cv2

from reachy_emotion import ui
from reachy_emotion.face_emotion import FaceEmotionDetector
from reachy_emotion.feedback import FeedbackGenerator
from reachy_emotion.fusion import EmotionFusion
from reachy_emotion.robot import NullRobot, ReachyRobot

WINDOW = "Reachy Emotion Prototype"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--robot", choices=["none", "reachy"], default="none",
                   help="'reachy' connects to a running reachy-mini-daemon (real robot or --sim)")
    p.add_argument("--robot-host", metavar="HOST",
                   help="Reachy Mini Wireless hostname or IP (default: this computer, then reachy-mini.local)")
    p.add_argument("--robot-media", action="store_true",
                   help="use the robot's camera and microphones instead of the laptop's")
    p.add_argument("--no-mic", action="store_true", help="skip speech; face only")
    p.add_argument("--mic", metavar="DEVICE",
                   help="microphone name (or part of it) or index, e.g. \"Reachy Mini Audio\" (default: system mic)")
    p.add_argument("--whisper", default="base", choices=["tiny", "base", "small"], help="Whisper model size")
    p.add_argument("--no-llm", action="store_true", help="use fixed hints instead of Ollama")
    p.add_argument("--speak", action="store_true", help="read feedback aloud")
    p.add_argument("--threshold", type=float, default=0.45,
                   help="below this confidence the light shows AMBER (default 0.45)")
    p.add_argument("--no-mirror", action="store_true",
                   help="don't mirror the webcam view (the robot's camera is never mirrored)")
    args = p.parse_args()
    if (args.robot_media or args.robot_host) and args.robot != "reachy":
        p.error("--robot-media and --robot-host need --robot reachy")
    if args.robot_media and args.mic:
        p.error("--mic picks a laptop/USB mic; with --robot-media the robot's mics are used")
    return args


class Webcam:
    def __init__(self, index):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open camera {index}. Is another app using it?")

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Camera stopped sending frames")
        return frame

    def close(self):
        self.cap.release()


class RobotCamera:
    """The Reachy Mini's head camera. read() returns None when no new frame
    has arrived yet — normal over Wi-Fi — and only gives up after a long gap."""

    def __init__(self, media, timeout=10):
        self.media = media
        self.timeout = timeout
        self._last_frame = time.time()

    def read(self):
        frame = self.media.get_frame()
        if frame is not None:
            self._last_frame = time.time()
        elif time.time() - self._last_frame > self.timeout:
            raise RuntimeError(f"No frames from the robot's camera for {self.timeout} s")
        return frame

    def close(self):
        pass


def main():
    # v1's EmotionPredictor prints symbols the default Windows console can't encode
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()

    detector = FaceEmotionDetector()
    fusion = EmotionFusion(confidence_threshold=args.threshold)
    feedback = FeedbackGenerator(use_llm=not args.no_llm, speak=args.speak)
    print(f"Feedback: {'Ollama ' + feedback.model if feedback.model else 'fixed hints'}")
    robot = ReachyRobot(host=args.robot_host, use_media=args.robot_media) if args.robot == "reachy" else NullRobot()
    if args.robot_media and robot.media.camera is None:
        robot.close()
        raise SystemExit("The robot's camera isn't available. Is another app using it?")

    mic = {"status": "off", "heard": None}
    listener = None
    if not args.no_mic:
        from reachy_emotion.speech import SpeechListener

        def on_speech(result):
            mic["heard"] = result["text"]
            fusion.update_speech(result)

        listener = SpeechListener(on_result=on_speech, on_status=lambda s: mic.update(status=s),
                                  whisper_model_size=args.whisper,
                                  device=int(args.mic) if args.mic and args.mic.isdigit() else args.mic,
                                  robot_media=robot.media if args.robot_media else None)
        listener.start()

    camera = RobotCamera(robot.media) if args.robot_media else Webcam(args.camera)
    # mirroring only feels natural when you're looking at yourself
    mirror = not (args.no_mirror or args.robot_media)

    cv2.namedWindow(WINDOW)
    try:
        while True:
            try:
                frame = camera.read()
            except RuntimeError as err:
                print(err)
                break
            if frame is None:
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue
            if mirror:
                frame = cv2.flip(frame, 1)
            if frame.shape[1] != 640:
                frame = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))

            face = detector.detect(frame)
            if face:
                fusion.update_face(face[1])
            reading = fusion.current()
            feedback.update(reading)
            robot.show(reading["light"] if reading else None)

            cv2.imshow(WINDOW, ui.render(frame, face, reading, feedback.text, mic["status"], mic["heard"]))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        camera.close()
        cv2.destroyAllWindows()
        if listener:
            listener.stop()
            listener.join(timeout=1)   # let it stop reading the robot's mics before they close
        robot.close()


if __name__ == "__main__":
    main()
