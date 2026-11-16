#!/usr/bin/env python3
"""
profile_loader.py  —  Tuner JSON → tuner_profile.json converter

Three usage modes:

  1. PASTE MODE (recommended — participant sends JSON, you paste it):
       python3 profile_loader.py P00
       python3 profile_loader.py P00 --profile "Watchful Companion"
     The script prompts you to paste the JSON, then press Enter twice.

  2. FILE MODE:
       python3 profile_loader.py tuner_export.json P00
       python3 profile_loader.py tuner_export.json P00 --profile "Ambient Presence"

  3. PIPE MODE:
       cat tuner_export.json | python3 profile_loader.py - P00
"""

import json, sys, os, argparse

PROFILE_MAP = {
    "Watchful Companion": {
        "calm_pref":   "Stay quietly attentive. Subtle, slow presence. This user finds comfort in knowing you are there.",
        "active_pref": "Stay steady and still. Low intensity almost always right. Do not add to their load.",
    },
    "Pattern Breaker": {
        "calm_pref":   "Be noticeable but not constant. Occasional clear gestures are welcome.",
        "active_pref": "More active — a clear, deliberate gesture can help break a thought loop. Higher intensity acceptable.",
    },
    "Ambient Presence": {
        "calm_pref":   "Barely move. Extremely quiet. Any movement that draws attention feels like intrusion.",
        "active_pref": "Go completely still or near-still. Any animation is unwanted. Idle is almost always correct.",
    },
}

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuner_profile.json")


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert Tuner JSON → tuner_profile.json for MiRo pipelines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # first positional: either a file path OR a participant ID (if no second arg given)
    p.add_argument("first",  help="JSON file path (or '-' for stdin), OR participant ID if pasting")
    p.add_argument("second", nargs="?", default=None, help="Participant ID (e.g. P00), if file given as first arg")
    p.add_argument("--profile", default=None,
                   choices=list(PROFILE_MAP.keys()),
                   help="Override auto-detected profile type")
    p.add_argument("--out", default=OUTPUT_PATH,
                   help="Output path (default: same folder as this script)")
    return p.parse_args()


def read_paste():
    """Interactive paste: collect lines until two consecutive blank lines."""
    print("\n" + "="*58)
    print("  Paste the JSON from the participant below.")
    print("  Press  Enter twice  when done.")
    print("="*58 + "\n")
    lines = []
    blank_streak = 0
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            blank_streak += 1
            if blank_streak >= 2:
                break
        else:
            blank_streak = 0
        lines.append(line)
    return "\n".join(lines)


def parse_json_text(text):
    """Parse JSON array, single object, or multiple objects separated by whitespace."""
    text = text.strip()
    if not text:
        print("ERROR: no JSON data received."); sys.exit(1)

    # try direct parse (array or single object)
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        pass

    # try extracting individual {...} objects (handles newline-separated records)
    records = []
    depth = 0
    buf   = []
    for ch in text:
        if ch == "{":
            depth += 1
        if depth > 0:
            buf.append(ch)
        if ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                candidate = "".join(buf).strip()
                try:
                    records.append(json.loads(candidate))
                except json.JSONDecodeError:
                    pass
                buf = []
    if records:
        return records

    print("ERROR: could not parse JSON. Check that you pasted the full content.")
    sys.exit(1)


def load_records_from_file(path):
    if path == "-":
        text = sys.stdin.read()
    else:
        if not os.path.exists(path):
            print(f"ERROR: file not found: {path}")
            print("Tip: run without a filename to use paste mode:")
            print(f"  python3 {os.path.basename(__file__)} <ParticipantID>")
            sys.exit(1)
        with open(path, encoding="utf-8") as f:
            text = f.read()
    return parse_json_text(text)


def guess_profile(calm_lma, active_lma):
    ca = active_lma
    if all(v == 0 for v in ca.values()):
        return "Ambient Presence"
    if ca.get("space_focus", 0) >= 70 and ca.get("weight_energy", 0) >= 30:
        return "Pattern Breaker"
    if ca.get("space_focus", 0) >= 50 and ca.get("weight_energy", 0) < 30:
        return "Watchful Companion"
    avg = sum(ca.values()) / max(len(ca), 1)
    return "Ambient Presence" if avg < 20 else "Watchful Companion"


def lma_to_robot(raw):
    return {
        "weight": raw.get("weight_energy",    50),
        "time":   raw.get("time_tempo",       50),
        "flow":   raw.get("flow_spontaneity", 50),
        "space":  raw.get("space_focus",      50),
    }


def find_records(records, pid):
    calm_rec   = None
    active_rec = None
    for r in records:
        if r.get("participantId") == pid:
            if r.get("contextMode") == "calm"   and calm_rec   is None:
                calm_rec   = r
            elif r.get("contextMode") == "active" and active_rec is None:
                active_rec = r
    return calm_rec, active_rec


def main():
    args = parse_args()

    # ── decide mode ─────────────────────────────────────────────────────────
    # If 'first' looks like a participant ID (starts with letter, no slash/dot),
    # treat as paste mode.
    looks_like_pid = (
        args.second is None
        and not os.sep in args.first
        and "." not in args.first
        and args.first != "-"
    )

    if looks_like_pid:
        pid  = args.first
        text = read_paste()
        records = parse_json_text(text)
    else:
        pid     = args.second or args.first  # fallback: only one arg means file=first, pid=?
        if args.second is None:
            print("ERROR: please provide participant ID as second argument.")
            print(f"  python3 {os.path.basename(__file__)} tuner_export.json P00")
            sys.exit(1)
        records = load_records_from_file(args.first)

    # ── find calm + active records ───────────────────────────────────────────
    calm_rec, active_rec = find_records(records, pid)

    if not calm_rec or not active_rec:
        found = sorted(set(r.get("participantId","?") for r in records))
        print(f"ERROR: could not find both calm + active records for '{pid}'")
        print(f"Participants found in pasted data: {found}")
        sys.exit(1)

    calm_lma   = lma_to_robot(calm_rec["lmaParameters"])
    active_lma = lma_to_robot(active_rec["lmaParameters"])
    profile_type = args.profile or guess_profile(calm_lma, active_lma)
    prefs = PROFILE_MAP[profile_type]

    out = {
        "participantId":  pid,
        "profileType":    profile_type,
        "mentalMetaphor": (
            calm_rec.get("mentalMetaphor","") + " / " +
            active_rec.get("mentalMetaphor","")
        ).strip(" /"),
        "calm_pref":      prefs["calm_pref"],
        "active_pref":    prefs["active_pref"],
        "lma_calm":       calm_lma,
        "lma_active":     active_lma,
        "_source_calm":   calm_rec.get("timestamp",""),
        "_source_active": active_rec.get("timestamp",""),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Profile written → {args.out}")
    print(f"   Participant:  {pid}")
    print(f"   Profile type: {profile_type}")
    print(f"   LMA calm:     W={calm_lma['weight']}  T={calm_lma['time']}  "
          f"F={calm_lma['flow']}  S={calm_lma['space']}")
    print(f"   LMA active:   W={active_lma['weight']}  T={active_lma['time']}  "
          f"F={active_lma['flow']}  S={active_lma['space']}")
    print(f"   Metaphor:     {out['mentalMetaphor']}\n")
    print("Next step:  start spontaneous_pipeline.py or scripted_pipeline.py (or experiment_ui.py)")


if __name__ == "__main__":
    main()
