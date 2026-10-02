# Reachy Emotion Prototype

Looks at a person through a camera, listens to what they say, and gives a child
a RED / AMBER / GREEN reading of how that person might be feeling — plus one
short, gentle hint about what to do. Built to run on a laptop webcam today and
move onto a [Reachy Mini](https://github.com/pollen-robotics/reachy_mini) later,
where the robot shows the reading with its head and antennas.

This is the next stage of `Sentiment-Analysis-Detection-v1` (it reuses that
project's text-emotion model) and borrows the local Ollama feedback idea from
`Sentiment-Guide-Chrome-Ext`.

## How it works

```
camera ─► face detection (OpenCV) ─► facial expression (HSEmotion, AffectNet) ─┐
                                                                               ├─► fused + smoothed ─► RED / AMBER / GREEN
mic ─► Whisper ─► text emotion (DistilRoBERTa, from v1) ───────────────────────┘          │
                                                         hint (Ollama if running, else fixed text)
                                                         virtual Reachy on screen / real Reachy pose
```

- Face scores are smoothed over time; speech counts for up to 40% and fades
  over 8 s after each sentence.
- The light only changes after the new colour holds for 1 s, so the robot
  doesn't twitch.
- Below the confidence threshold the light is AMBER and the hint says so.
- If the face and the words confidently disagree (e.g. smiling while saying
  "I hate you"), the light is AMBER with a "mixed signals" hint — sarcasm and
  masking are exactly what's hardest to read.

## Setup (Windows)

The code expects `Sentiment-Analysis-Detection-v1` to sit next to this folder.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -r requirements.txt
```

The first run downloads the face model (~16 MB), Whisper `base` (~140 MB) and
the text-emotion model (~330 MB).

Optional: install [Ollama](https://ollama.com/download) and `ollama pull gemma3:4b`
for model-written hints. Without it, fixed hints are used. The first load of
the model can take a few minutes on a laptop; fixed hints show until it's ready,
then it stays in memory for 30 minutes and replies in a few seconds.

## Run

```powershell
.\.venv\Scripts\python main.py              # webcam + mic
.\.venv\Scripts\python main.py --no-mic     # face only
.\.venv\Scripts\python main.py --speak      # read hints aloud
.\.venv\Scripts\python main.py --robot reachy --robot-media  # on a Reachy Mini (see below)
.\.venv\Scripts\python main.py --help       # all options
```

Press `q` or `Esc` in the window to quit.

## Running on a Reachy Mini

Install the SDK once (it includes the MuJoCo simulator too):

```powershell
.\.venv\Scripts\python -m pip install "reachy-mini[mujoco]"
```

**Reachy Mini Lite (USB).** The daemon runs on this laptop. Plug the robot in
and power it on, then:

```powershell
.\.venv\Scripts\python -m reachy_mini.daemon.app.main          # terminal 1
.\.venv\Scripts\python main.py --robot reachy --robot-media   # terminal 2
```

**Reachy Mini Wireless.** The daemon already runs on the robot. Put the laptop
on the same Wi-Fi and skip terminal 1:

```powershell
.\.venv\Scripts\python main.py --robot reachy --robot-media
.\.venv\Scripts\python main.py --robot reachy --robot-media --robot-host 192.168.1.42  # if reachy-mini.local doesn't resolve
```

`--robot-media` makes the app see through the robot's head camera and hear
through its microphones, so the robot faces the person being read. Leave it off
to keep using the laptop webcam and mic while the robot only moves. The robot's
camera view is never mirrored. Hints play on the laptop speakers (`--speak`), so
the child hears them and the other person doesn't.

On a Lite, if the robot's camera won't stream through the SDK, open it as a
normal USB camera instead. Without `--robot-media` the app tells the daemon to
release the camera and mic:

```powershell
.\.venv\Scripts\python main.py --robot reachy --camera 1 --mic "Reachy Mini Audio"
```

Try `--camera 1`, `2`, and so on until you see the robot's view.

On start the motors are switched on. When you quit, the robot goes to its
sleep pose, so it's safe to power off.

### Testing without a robot (MuJoCo simulator)

```powershell
.\.venv\Scripts\python -m reachy_mini.daemon.app.main --sim    # terminal 1 — opens a 3D window
.\.venv\Scripts\python main.py --robot reachy  # terminal 2
```

The simulated robot moves with the reading while your webcam and mic do the
sensing. Tune the poses in `POSES` in `reachy_emotion/robot.py`. (On Windows
the simulator's camera doesn't stream to the SDK, so test `--robot-media` on the
real robot.)

## Files

| File | Purpose |
|---|---|
| `main.py` | Camera loop and command-line options |
| `reachy_emotion/face_emotion.py` | Face detection + expression scores |
| `reachy_emotion/speech.py` | Background mic → Whisper → text emotion |
| `reachy_emotion/fusion.py` | Combines, smooths and steadies the reading |
| `reachy_emotion/feedback.py` | Hint text (Ollama or fixed), optional speech |
| `reachy_emotion/robot.py` | Pose per light; Reachy Mini driver and microphone |
| `reachy_emotion/ui.py` | Window with camera, bars and virtual Reachy |

## Limits and privacy

- Facial expression models are trained mostly on posed adult faces and are much
  less reliable across ages, cultures and neurotypes. Treat readings as a hint,
  never a fact. The hint wording is deliberately hedged.
- Nothing is saved: frames and audio are processed in memory and discarded.
- This is for testing with consenting people only. Any study with children
  needs ethics (HREC) clearance first.

## Attribution
Builds on work originally created by Evangeline Tanoto, Semester 1 2026, under
the supervision of Hermann Tsai and Susannah Soon, Curtin University.
Contributions and continued development by Lee Chao Cherng.
