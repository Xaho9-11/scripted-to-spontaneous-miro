# From Scripted to Spontaneous — VLM-driven behaviour for a desktop companion robot

Code for **"From Scripted to Spontaneous: Personalising Animacy of VLM-Driven Desktop Companion Robot Behaviours
During Focused Work"**, accepted at ACM TEI 2027 (citation and DOI will be added when the proceedings are out).

A desktop companion robot should feel alive without pulling attention away from the person working next to it.
This project built a behaviour system for the [MiRo-E](https://www.miro-e.com/) robot in which a
vision-language model (VLM) interprets the live context — the person's posture, gaze and touch — and selects
and parameterises actions from a bounded repertoire, shaped by a movement profile that the user set beforehand.

The study had two phases:

1. **Co-design (N = 8).** Participants used the *Metaphor Tuner* to configure the movement quality they would like
   from a robot during calm and during demanding work, along the four Effort dimensions of Laban Movement Analysis
   (Weight, Time, Flow, Space), and described their choices through movement metaphors. Three behaviour profiles
   were derived from this.
2. **Evaluation (N = 17, within-subject).** The **Spontaneous** system (VLM selection with the participant's profile
   and live context) was compared with a **Scripted** system that played the same fixed sequence to every
   participant, on perceived animacy, extraneous cognitive load and discomfort.

## System

```
                     experiment_ui.py (operator console: start / switch / stop a condition, live views)
                       │ starts one of
MiRo-E camera ─┐       ▼                                                ┌─▶ neck, ears, tail, eyelids, lights
touch sensors ─┼─▶ spontaneous_pipeline.py ─┐                           │
               │   MediaPipe posture + gaze  ├─/miro/behavior_cmd─▶ miro_director.py ─▶ MiRo-E (ROS, MDK)
               │   VLM chooses the action    │
touch sensors ─┴─▶ scripted_pipeline.py ────┘
                   fixed sequence, fixed interval, no camera or VLM
```

Both conditions share the same action library (twelve primitives), the same director and the same touch
reactions; they differ in how the next action is chosen.

- **`robot/spontaneous_pipeline.py`** — the Spontaneous condition. Reads MiRo's camera and touch sensors over ROS,
  extracts posture and gaze with MediaPipe against a per-participant baseline, and asks the VLM (Gemini) to choose
  one action together with its intensity and duration. The prompt carries the participant's profile and LMA values
  (from `tuner_profile.json`) and a diversity rule; a code-level fallback replaces invalid answers. Touch and eye
  contact trigger reflexes without waiting for the VLM.
- **`robot/scripted_pipeline.py`** — the Scripted condition. A time-driven state machine (alert → calm → lazy →
  drowsy) plays a fixed action sequence at a fixed interval, the same for every participant, with fixed touch
  responses.
- **`robot/miro_director.py`** — the animatronics engine shared by both conditions. Renders each action as smooth
  joint, ear, tail, eyelid and light trajectories at 50 Hz, with fade-outs, a sleep chain (drowsy → sleep → wake)
  with timers, and a background layer of small life-like motion (`noise_mode`: `organic` or `mechanical`).
- **`robot/experiment_ui.py`** — the operator console (PyQt): starts and switches the condition programs (keys
  S / M / X), shows the external and robot-eye cameras and the live state of the pipeline.
- **`robot/profile_loader.py`** — turns a participant's Metaphor Tuner export into `tuner_profile.json`
  (format: `tuner_profile.example.json`).
- **`codesign/metaphor_tuner.html`** — the Metaphor Tuner used on an iPad in Phase 1: four LMA sliders, calm/active
  context, metaphor tags, and a live abstract creature whose motion follows the settings; exports the result as JSON.
- **`analysis/`** — statistics and figures for the paper (paired tests on Godspeed animacy, non-inferiority tests on
  cognitive load and discomfort, tuner summaries).

## Running

Requirements: a MiRo-E with the MiRo Developer Kit (ROS) on the robot, and a PC on the same network with Python 3,
`rospy` (from the MDK), `opencv-python`, `mediapipe`, `numpy`, `google-genai`, `PyQt5`, and a USB camera for the
operator's external view.

```bash
export GEMINI_API_KEY=...                                     # your own key; never commit it
python3 robot/profile_loader.py P00                           # paste the participant's Tuner export
MIRO_IP=<robot> PC_IP=<pc> bash robot/run.sh director         # terminal 1, keep running
MIRO_IP=<robot> PC_IP=<pc> bash robot/run.sh ui               # terminal 2: S = Spontaneous, M = Scripted, X = stop
```

The Metaphor Tuner is a single HTML file; open it in a browser (it loads Tailwind from a CDN).

## Data

No study data is included. The questionnaire workbook, the tuner exports and the interview material contain
participant responses and stay with the research team; the analysis scripts expect them under `data/` (see
[analysis/README.md](analysis/README.md)).

## Citation

Chenghao Wang, Chao Wang, Jing Li, and Emilia I. Barakova. 2027. From Scripted to Spontaneous: Personalising
Animacy of VLM-Driven Desktop Companion Robot Behaviours During Focused Work. In *Proceedings of the 21st
International Conference on Tangible, Embedded, and Embodied Interaction (TEI '27)*. ACM. (DOI to follow.)

Developed with the help of AI coding assistants.

## License

MIT, see [LICENSE](LICENSE). MiRo-E, its MDK and the other dependencies keep their own licences
([THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).
