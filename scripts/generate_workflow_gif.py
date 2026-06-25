from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "assets" / "aiterate-workflow.gif"
WIDTH = 980
HEIGHT = 560

STEPS = [
    ("Import context", "Upload notes, data files, policies, examples, and an optional baseline."),
    ("Weight policies", "Set priorities so AI artifact changes become measurable rules."),
    ("Configure models", "Choose optimizer and target providers, with separate credentials if needed."),
    ("Run optimization", "Generate accepted and rejected artifact versions with lineage."),
    ("Evaluate regressions", "Check grounding, JSON shape, similarity, safety, PII, and policy coverage."),
    ("Approve best version", "Review insights, compare versions, and approve the release candidate."),
    ("Promote with Git", "Create a PR and keep a traceable artifact history."),
]

COLORS = {
    "bg": "#f5f8fc",
    "ink": "#111827",
    "muted": "#64748b",
    "brand": "#155eef",
    "brand_dark": "#0f3b96",
    "success": "#15803d",
    "line": "#dbe4f0",
    "panel": "#ffffff",
    "nav": "#0f172a",
    "nav_card": "#1e293b",
    "nav_done": "#123d31",
}


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frames = [draw_frame(index) for index in range(len(STEPS))]
    frames[0].save(
        OUTPUT,
        save_all=True,
        append_images=frames[1:],
        duration=1050,
        loop=0,
        disposal=2,
        optimize=False,
    )
    print(f"Wrote {OUTPUT}")


def draw_frame(active_index: int) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), COLORS["bg"])
    draw = ImageDraw.Draw(image)
    font_big = font(46, bold=True)
    font_title = font(28, bold=True)
    font_body = font(19)
    font_small = font(15, bold=True)

    draw.rounded_rectangle((22, 22, WIDTH - 22, 132), radius=10, fill=COLORS["panel"], outline=COLORS["line"], width=2)
    draw.text((52, 44), "AIterate", font=font_big, fill=COLORS["ink"])
    draw.text(
        (52, 96),
        "Governed AI artifact lifecycle and change management from raw data and policies",
        font=font_body,
        fill=COLORS["muted"],
    )

    draw.rounded_rectangle((22, 154, 278, HEIGHT - 24), radius=10, fill=COLORS["nav"])
    draw.text((44, 176), "SETUP PROGRESS", font=font_small, fill="#cbd5e1")

    for index, (title, _) in enumerate(STEPS):
        y = 210 + index * 44
        done = index < active_index
        active = index == active_index
        fill = COLORS["nav_done"] if done else COLORS["nav_card"]
        outline = "#36b37e" if done else "#334155"
        if active:
            outline = COLORS["brand"]
        draw.rounded_rectangle((42, y, 258, y + 34), radius=8, fill=fill, outline=outline, width=2)
        badge = "#" if not done else "✓"
        badge_text = str(index + 1) if not done else badge
        badge_fill = COLORS["success"] if done else "#475569"
        if active:
            badge_fill = COLORS["brand"]
        draw.ellipse((54, y + 5, 78, y + 29), fill=badge_fill)
        draw.text((62 if len(badge_text) == 1 else 59, y + 9), badge_text, font=font_small, fill="#ffffff")
        draw.text((90, y + 8), title, font=font_small, fill="#ffffff")

    card_x = 310
    card_y = 164
    draw.rounded_rectangle((card_x, card_y, WIDTH - 32, HEIGHT - 34), radius=10, fill=COLORS["panel"], outline=COLORS["line"], width=2)
    title, subtitle = STEPS[active_index]
    draw.text((card_x + 34, card_y + 34), title, font=font_title, fill=COLORS["ink"])
    draw.text((card_x + 34, card_y + 74), subtitle, font=font_body, fill=COLORS["muted"])

    draw_progress(draw, card_x + 34, card_y + 128, active_index)
    draw_demo_panel(draw, card_x + 34, card_y + 188, active_index)

    return image


def draw_progress(draw: ImageDraw.ImageDraw, x: int, y: int, active_index: int) -> None:
    dot_gap = 82
    for index in range(len(STEPS)):
        cx = x + index * dot_gap
        if index < len(STEPS) - 1:
            line_fill = COLORS["brand"] if index < active_index else COLORS["line"]
            draw.line((cx + 14, y, cx + dot_gap - 14, y), fill=line_fill, width=5)
        fill = COLORS["brand"] if index <= active_index else "#e2e8f0"
        draw.ellipse((cx - 12, y - 12, cx + 12, y + 12), fill=fill)


def draw_demo_panel(draw: ImageDraw.ImageDraw, x: int, y: int, active_index: int) -> None:
    font_body = font(18)
    font_code = font(15)
    blocks = {
        0: [
            ("raw_support_notes.txt", "#ffffff"),
            ("Refund tickets, policy notes, escalation examples", "#eef4ff"),
            ("Detected: policy rules, user scenarios, citation requirements", "#e8f7ee"),
        ],
        1: [
            ("accuracy 0.35", "#eef4ff"),
            ("citations 0.30", "#eef4ff"),
            ("uncertainty escalation 0.35", "#eef4ff"),
        ],
        2: [
            ("Optimizer: OpenAI / gpt-4.1", "#eef4ff"),
            ("Target: Anthropic / Claude", "#fff4e6"),
            ("Credentials: saved separately, never displayed", "#e8f7ee"),
        ],
        3: [
            ("Accepted v1 score 0.74", "#e8f7ee"),
            ("Rejected candidate score 0.69", "#fff4e6"),
            ("Accepted v2 score 0.88", "#e8f7ee"),
        ],
        4: [
            ("Eval score 0.91 | pass rate 92%", "#e8f7ee"),
            ("PASS source_grounded, uncertainty_handling", "#e8f7ee"),
            ("FAIL brevity: tighten final artifact", "#fff4e6"),
        ],
        5: [
            ("Review best version", "#eef4ff"),
            ("Artifact changes needed: tighten brevity", "#fff4e6"),
            ("Approved for promotion", "#e8f7ee"),
        ],
        6: [
            ("Create Git PR", "#eef4ff"),
            ("Attach eval report and lineage", "#e8f7ee"),
            ("Ready for production review", "#e8f7ee"),
        ],
    }
    for idx, (text, fill) in enumerate(blocks[active_index]):
        top = y + idx * 56
        draw.rounded_rectangle((x, top, WIDTH - 76, top + 42), radius=8, fill=fill, outline=COLORS["line"])
        draw.text((x + 18, top + 11), text, font=font_code if idx else font_body, fill=COLORS["ink"])


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


if __name__ == "__main__":
    main()
