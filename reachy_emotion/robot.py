"""How the robot shows the traffic light with its body.

POSES is the single source of truth: the on-screen virtual Reachy and the real
(or simulated) Reachy Mini both read from it. Tune the numbers in the MuJoCo
sim — angles are degrees, z is millimetres.

Antennas use the SDK's order, (right, left). Negative right / positive left
splays them outward into a V; the opposite signs cross them into an X.
"""

import queue
import threading

import numpy as np

POSES = {
    # nobody there: relaxed, looking ahead
    None:    {"roll": 0,  "pitch": 0,   "z": 0,   "antennas": (0, 0)},
    # friendly: head up a little, antennas perked into a V
    "GREEN": {"roll": 0,  "pitch": -10, "z": 8,   "antennas": (-35, 35)},
    # unsure: curious head tilt, one antenna up and one out
    "AMBER": {"roll": 15, "pitch": 0,   "z": 0,   "antennas": (-5, 45)},
    # careful: head dips and pulls back, antennas droop out to the sides
    "RED":   {"roll": 0,  "pitch": 15,  "z": -10, "antennas": (-100, 100)},
}


class NullRobot:
    """No hardware — the on-screen virtual Reachy is the only output"""

    def show(self, light):
        pass

    def close(self):
        pass


class ReachyRobot:
    """Drives a Reachy Mini (real or `reachy-mini-daemon --sim`) in a worker
    thread so slow moves never stall the camera loop.

    host: None tries a daemon on this computer first (Reachy Mini Lite over USB,
    or the sim), then `reachy-mini.local` (Reachy Mini Wireless). Pass a
    hostname or IP to go straight to a Wireless robot.
    use_media: True uses the robot's own camera and microphones via
    `self.media`; False leaves them alone and the laptop's are used.
    """

    def __init__(self, host=None, use_media=False, move_seconds=0.8):
        from reachy_mini import ReachyMini
        from reachy_mini.utils import create_head_pose

        self._create_head_pose = create_head_pose
        self.move_seconds = move_seconds
        connection = {"host": host, "connection_mode": "network"} if host else {}
        self.mini = ReachyMini(media_backend="default" if use_media else "no_media", **connection)
        self.media = self.mini.media if use_media else None
        # a real robot can come up limp (e.g. after goto_sleep); the sim ignores this
        self.mini.enable_motors()
        self._targets = queue.Queue()
        self._last = object()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def show(self, light):
        if light != self._last:
            self._last = light
            self._targets.put(light)

    def _worker(self):
        while True:
            light = self._targets.get()
            if light == "STOP":
                return
            while not self._targets.empty():     # skip to the newest target
                light = self._targets.get_nowait()
                if light == "STOP":
                    return
            pose = POSES[light]
            try:
                self.mini.goto_target(
                    head=self._create_head_pose(z=pose["z"], roll=pose["roll"], pitch=pose["pitch"],
                                                degrees=True, mm=True),
                    antennas=np.deg2rad(pose["antennas"]),
                    duration=self.move_seconds,
                )
            except Exception as err:  # keep the app running if the daemon drops
                print("Robot: move failed —", err)

    def close(self):
        self._targets.put("STOP")
        self._thread.join(timeout=self.move_seconds + 2)   # don't send two moves at once
        try:
            # settle into the SDK's resting pose so a real robot is safe to switch off
            self.mini.goto_sleep()
        except Exception as err:
            print("Robot: could not go to sleep —", err)
        finally:
            self.mini.__exit__(None, None, None)


class RobotMic:
    """The Reachy Mini's microphones as a drop-in for the laptop mic: feeds
    mono blocks of `block` samples at `rate` Hz into `blocks`."""

    def __init__(self, media, rate, block):
        self.media = media
        self.rate = rate
        self.block = block
        self._stop = threading.Event()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self.media.stop_recording()

    def open(self, blocks):
        self.media.start_recording()
        threading.Thread(target=self._pump, args=(blocks,), daemon=True).start()
        return self

    def _pump(self, blocks):
        in_rate = self.media.get_input_audio_samplerate()
        pending = np.zeros(0, np.float32)
        while not self._stop.is_set():
            chunk = self.media.get_audio_sample()       # waits up to 20 ms
            if chunk is None:
                continue
            mono = chunk.mean(axis=1) if chunk.ndim == 2 else chunk
            if in_rate != self.rate:
                n = int(len(mono) * self.rate / in_rate)
                mono = np.interp(np.linspace(0, len(mono), n, endpoint=False), np.arange(len(mono)), mono)
            pending = np.concatenate([pending, mono.astype(np.float32)])
            # the speech detector counts fixed-size blocks, so keep them the laptop's size
            while len(pending) >= self.block:
                blocks.put(pending[:self.block])
                pending = pending[self.block:]
