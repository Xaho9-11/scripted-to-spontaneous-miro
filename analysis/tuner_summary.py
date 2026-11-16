"""Recompute Phase 1 Tuner summaries and the box-plus-paired-point figure."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "phase1_tuner.json"  # tuner exports of all participants; not published
FIGURE_PATH = ROOT / "figures" / "tuner_boxplot.png"
PROFILE_FIGURE_PATH = ROOT / "figures" / "profile_quadrants.png"
DIMENSIONS = ("weight", "time", "flow", "space")


def draw_boxplot(
    draw: ImageDraw.ImageDraw,
    values: np.ndarray,
    x: int,
    plot_top: int,
    plot_bottom: int,
    outline: str,
    fill: str,
) -> None:
    """Draw a compact min/max boxplot behind the paired observations."""

    def y(value: float) -> int:
        return plot_bottom - int((value / 100) * (plot_bottom - plot_top))

    q1, median, q3 = np.percentile(values, [25, 50, 75])
    low, high = float(values.min()), float(values.max())
    half_box = 38
    half_cap = 21
    draw.line((x, y(high), x, y(low)), fill=outline, width=3)
    draw.line((x - half_cap, y(high), x + half_cap, y(high)), fill=outline, width=3)
    draw.line((x - half_cap, y(low), x + half_cap, y(low)), fill=outline, width=3)
    draw.rectangle(
        (x - half_box, y(q3), x + half_box, y(q1)),
        fill=fill,
        outline=outline,
        width=4,
    )
    draw.line(
        (x - half_box, y(median), x + half_box, y(median)),
        fill=outline,
        width=4,
    )


def main() -> None:
    records = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    participants = sorted({record["participant_id"] for record in records})
    lookup = {
        (record["participant_id"], record["context_mode"]): record
        for record in records
    }

    if len(records) != 16 or len(participants) != 8:
        raise ValueError("Expected 16 records from eight participants")
    for participant in participants:
        for mode in ("Calm", "Active"):
            if (participant, mode) not in lookup:
                raise ValueError(f"Missing {mode} record for {participant}")

    width, height = 1800, 1250
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    bold_path = Path("C:/Windows/Fonts/arialbd.ttf")
    body = ImageFont.truetype(str(font_path), 32)
    small = ImageFont.truetype(str(font_path), 27)
    title = ImageFont.truetype(str(bold_path), 42)
    panel_title = ImageFont.truetype(str(bold_path), 34)
    calm_colour = "#2878B5"
    active_colour = "#D84A4A"
    calm_fill = "#DCEBF6"
    active_fill = "#F6DDDD"
    line_colour = "#A5AAB0"
    grid_colour = "#D9D9D9"
    draw.text((width / 2, 25), "Participant-level Metaphor Tuner settings", font=title, fill="#222222", anchor="ma")

    panels = [
        (110, 105, 850, 590),
        (950, 105, 1690, 590),
        (110, 690, 850, 1175),
        (950, 690, 1690, 1175),
    ]

    for panel, dimension in zip(panels, DIMENSIONS):
        calm = np.array([lookup[(p, "Calm")][dimension] for p in participants])
        active = np.array([lookup[(p, "Active")][dimension] for p in participants])
        left, top, right, bottom = panel
        plot_left, plot_top = left + 90, top + 70
        plot_right, plot_bottom = right - 35, bottom - 75
        x_calm = plot_left + int((plot_right - plot_left) * 0.30)
        x_active = plot_left + int((plot_right - plot_left) * 0.70)

        draw.text(((left + right) / 2, top + 8), dimension.capitalize(), font=panel_title, fill="#222222", anchor="ma")
        for tick in range(0, 101, 20):
            y = plot_bottom - int((tick / 100) * (plot_bottom - plot_top))
            draw.line((plot_left, y, plot_right, y), fill=grid_colour, width=2)
            draw.text((plot_left - 18, y), str(tick), font=small, fill="#444444", anchor="rm")
        draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#333333", width=3)
        draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#333333", width=3)

        draw_boxplot(draw, calm, x_calm, plot_top, plot_bottom, calm_colour, calm_fill)
        draw_boxplot(draw, active, x_active, plot_top, plot_bottom, active_colour, active_fill)

        jitter = np.linspace(-25, 25, len(participants)).astype(int)
        for calm_value, active_value, offset in zip(calm, active, jitter):
            y_calm = plot_bottom - int((calm_value / 100) * (plot_bottom - plot_top))
            y_active = plot_bottom - int((active_value / 100) * (plot_bottom - plot_top))
            point_calm = x_calm + int(offset)
            point_active = x_active + int(offset)
            draw.line((point_calm, y_calm, point_active, y_active), fill=line_colour, width=3)
            radius = 9
            draw.ellipse(
                (point_calm - radius, y_calm - radius, point_calm + radius, y_calm + radius),
                fill=calm_colour,
                outline="white",
                width=2,
            )
            draw.ellipse(
                (point_active - radius, y_active - radius, point_active + radius, y_active + radius),
                fill=active_colour,
                outline="white",
                width=2,
            )
        draw.text((x_calm, plot_bottom + 25), "Calm", font=body, fill="#333333", anchor="ma")
        draw.text((x_active, plot_bottom + 25), "Active", font=body, fill="#333333", anchor="ma")
        print(
            f"{dimension}: Calm mean={calm.mean():.2f}, SD={calm.std(ddof=1):.2f}; "
            f"Active mean={active.mean():.2f}, SD={active.std(ddof=1):.2f}; "
            f"mean change={(active - calm).mean():.2f}"
        )

    rotated_label = Image.new("RGBA", (500, 60), (255, 255, 255, 0))
    rotated_draw = ImageDraw.Draw(rotated_label)
    rotated_draw.text((250, 30), "Tuner value (0–100)", font=body, fill="#333333", anchor="mm")
    rotated_label = rotated_label.rotate(90, expand=True)
    image.paste(rotated_label, (5, 365), rotated_label)
    FIGURE_PATH.parent.mkdir(exist_ok=True)
    image.save(FIGURE_PATH, dpi=(300, 300))

    if "--boxplot-only" in sys.argv:
        return

    active_rows = [lookup[(participant, "Active")] for participant in participants]
    map_width, map_height = 1800, 1100
    profile_image = Image.new("RGB", (map_width, map_height), "white")
    profile_draw = ImageDraw.Draw(profile_image)
    map_left, map_top, map_right, map_bottom = 190, 90, 1690, 900
    mid_x = map_left + (map_right - map_left) // 2
    mid_y = map_bottom - (map_bottom - map_top) // 2
    profile_draw.rectangle((map_left, map_top, mid_x, mid_y), fill="#F7F7F7")
    profile_draw.rectangle((mid_x, map_top, map_right, mid_y), fill="#FCEEEE")
    profile_draw.rectangle((map_left, mid_y, mid_x, map_bottom), fill="#EEF3FA")
    profile_draw.rectangle((mid_x, mid_y, map_right, map_bottom), fill="#FFF4EA")
    for tick in range(0, 101, 20):
        x = map_left + int((tick / 100) * (map_right - map_left))
        y = map_bottom - int((tick / 100) * (map_bottom - map_top))
        profile_draw.line((x, map_top, x, map_bottom), fill="#DDDDDD", width=2)
        profile_draw.line((map_left, y, map_right, y), fill="#DDDDDD", width=2)
        profile_draw.text((x, map_bottom + 18), str(tick), font=small, fill="#444444", anchor="ma")
        profile_draw.text((map_left - 18, y), str(tick), font=small, fill="#444444", anchor="rm")
    profile_draw.line((mid_x, map_top, mid_x, map_bottom), fill="#888888", width=4)
    profile_draw.line((map_left, mid_y, map_right, mid_y), fill="#888888", width=4)
    profile_draw.line((map_left, map_top, map_left, map_bottom), fill="#333333", width=4)
    profile_draw.line((map_left, map_bottom, map_right, map_bottom), fill="#333333", width=4)

    profile_draw.text((map_right - 20, map_top + 20), "Pattern Breaker", font=panel_title, fill="#B33F45", anchor="ra")
    profile_draw.text((map_left + 20, map_top + 20), "Unpopulated", font=panel_title, fill="#777777", anchor="la")
    profile_draw.text((map_left + 20, mid_y + 20), "Ambient Presence", font=panel_title, fill="#3569A8", anchor="la")
    profile_draw.text((map_right - 20, mid_y + 20), "Watchful Companion", font=panel_title, fill="#C76C2C", anchor="ra")

    colours = {
        "Pattern Breaker": "#C84F55",
        "Ambient Presence": "#4C72B0",
        "Watchful Companion": "#DD8452",
    }
    coincident_labels: dict[tuple[int, int], list[str]] = {}
    for row in active_rows:
        energy = (row["weight"] + row["time"]) / 2
        if row["space"] >= 50 and energy >= 50:
            profile = "Pattern Breaker"
        elif row["space"] >= 50:
            profile = "Watchful Companion"
        else:
            profile = "Ambient Presence"
        x = map_left + int((row["space"] / 100) * (map_right - map_left))
        y = map_bottom - int((energy / 100) * (map_bottom - map_top))
        radius = 14 + int(row["flow"] * 0.28)
        profile_draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=colours[profile],
            outline="white",
            width=4,
        )
        coincident_labels.setdefault((x, y), []).append(row["participant_id"])
        print(
            f"{row['participant_id']}: Energy={energy:.1f}, Space={row['space']}, "
            f"Flow={row['flow']}, profile={profile}"
        )
    for (x, y), labels in coincident_labels.items():
        for index, label in enumerate(labels):
            label_y = max(map_top + 55, y - 38 - index * 34)
            profile_draw.text((x, label_y), label, font=small, fill="#222222", anchor="ms")

    profile_draw.text(((map_left + map_right) / 2, map_height - 55), "Active-state Space (0–100: indirect to direct)", font=body, fill="#222222", anchor="mm")
    energy_label = Image.new("RGBA", (760, 60), (255, 255, 255, 0))
    energy_draw = ImageDraw.Draw(energy_label)
    energy_draw.text((380, 30), "Active-state Energy (mean of Weight and Time)", font=body, fill="#222222", anchor="mm")
    energy_label = energy_label.rotate(90, expand=True)
    profile_image.paste(energy_label, (20, 170), energy_label)
    profile_image.save(PROFILE_FIGURE_PATH, dpi=(300, 300))


if __name__ == "__main__":
    main()
