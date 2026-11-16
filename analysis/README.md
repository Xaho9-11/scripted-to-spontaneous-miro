# Analysis

Scripts behind the statistics and figures in the paper. The data are **not** in this repository (they contain
participant responses); the scripts expect them here:

| File | Content |
|---|---|
| `data/phase2_data.xlsx` | Phase 2 workbook, sheet "Data Entry" (header on row 3): one row per participant with the Klepsch ECL, Godspeed animacy and RoSAS discomfort items for both conditions, and the participant's behaviour profile. |
| `data/phase1_tuner.json` | All Metaphor Tuner exports of Phase 1: participant code, metaphor tags, context (Calm / Active) and the four LMA values (0–100). |

| Script | What it does |
|---|---|
| `questionnaire.py [xlsx]` | Per-scale means and SDs for both conditions, paired t-tests and non-inferiority tests; writes `results_summary.txt`. |
| `study_stats.py [xlsx]` | H1 paired t-test on Godspeed animacy; H2 non-inferiority tests on cognitive load and discomfort. |
| `figures.py [xlsx]` | H1 bar and paired plots, H2 non-inferiority plot. |
| `manuscript_stats.py [xlsx]` | Recomputes the numbers reported in the manuscript and draws the paired-score and non-inferiority figures into `figures/`. |
| `tuner_summary.py` | Phase 1 tuner summaries, the LMA box plot and the profile quadrant figure (`figures/`). |

Python 3 with `pandas`, `numpy`, `scipy`, `matplotlib`, `openpyxl` and `pillow`. Two scripts draw with Arial from
`C:/Windows/Fonts`; change the font path on other systems.
