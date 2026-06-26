from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "assets" / "aiterate-workflow.gif"
WIDTH = 980
HEIGHT = 560

STEPS = [
    ("Run History", "Open approved artifacts, continue from their best version, or compare models."),
    ("Import context", "Upload examples, policies, references, and an optional baseline artifact."),
    ("Weight policies", "Choose train/test split and turn policies into measurable scoring rules."),
    ("Configure models", "Select optimizer and target models, with separate credentials when needed."),
    ("Tracking optional", "Send runs to MLflow or LangSmith, or keep the workflow local."),
    ("Run optimizer", "Generate scored candidates and keep complete experiment lineage."),
    ("Review and approve", "See score progress, diffs, attempts not used, and approve the best version."),
    ("Create Git PR", "Persist project Git settings, create a PR, or export the full package."),
    ("Compare models", "Run the same approved prompt against selected models and eval modes."),
]
STEP_BADGES = ["H", "1", "2", "3", "4", "5", "6", "7", "8"]

COLORS = {
    "bg": "#f5f8fc",
    "ink": "#111827",
    "muted": "#64748b",
    "brand": "#155eef",
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

    draw.rounded_rectangle(
        (22, 22, WIDTH - 22, 132),
        radius=10,
        fill=COLORS["panel"],
        outline=COLORS["line"],
        width=2,
    )
    draw.text((52, 44), "Aiterate", font=font_big, fill=COLORS["ink"])
    draw.text(
        (52, 96),
        "Governed AI artifact lifecycle and change management from raw data and policies",
        font=font_body,
        fill=COLORS["muted"],
    )

    draw.rounded_rectangle((22, 154, 278, HEIGHT - 24), radius=10, fill=COLORS["nav"])
    draw.text((44, 176), "WORKSPACE", font=font_small, fill="#cbd5e1")

    for index, (title, _) in enumerate(STEPS):
        y = 204 + index * 32
        done = index < active_index
        active = index == active_index
        fill = COLORS["nav_done"] if done else COLORS["nav_card"]
        outline = "#36b37e" if done else "#334155"
        if active:
            outline = COLORS["brand"]
        draw.rounded_rectangle((42, y, 258, y + 27), radius=8, fill=fill, outline=outline, width=2)
        badge_text = STEP_BADGES[index] if not done else "OK"
        badge_fill = COLORS["success"] if done else "#475569"
        if active:
            badge_fill = COLORS["brand"]
        draw.ellipse((52, y + 4, 76, y + 28), fill=badge_fill)
        draw.text((61 if len(badge_text) == 1 else 56, y + 8), badge_text, font=font_small, fill="#ffffff")
        draw.text((88, y + 5), title, font=font_small, fill="#ffffff")

    card_x = 310
    card_y = 164
    draw.rounded_rectangle(
        (card_x, card_y, WIDTH - 32, HEIGHT - 34),
        radius=10,
        fill=COLORS["panel"],
        outline=COLORS["line"],
        width=2,
    )
    title, subtitle = STEPS[active_index]
    draw.text((card_x + 34, card_y + 34), title, font=font_title, fill=COLORS["ink"])
    draw.text((card_x + 34, card_y + 74), subtitle, font=font_body, fill=COLORS["muted"])

    draw_progress(draw, card_x + 34, card_y + 128, active_index)
    draw_demo_panel(draw, card_x + 34, card_y + 188, active_index)

    return image


def draw_progress(draw: ImageDraw.ImageDraw, x: int, y: int, active_index: int) -> None:
    dot_gap = 62
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
            ("Approved artifact badge in Run History", "#e8f7ee"),
            ("Open run review or delete with confirmation", "#eef4ff"),
            ("Best version becomes the next baseline", "#fff4e6"),
        ],
        1: [
            ("Data / Examples: tickets, conversations, eval rows", "#eef4ff"),
            ("Policies: rules, tone, compliance, acceptance criteria", "#fff4e6"),
            ("Knowledge Base: docs, SOPs, manuals, source references", "#e8f7ee"),
        ],
        2: [
            ("Train split 80% | Test holdout 20%", "#eef4ff"),
            ("citations 0.30 | escalation 0.35", "#fff4e6"),
            ("grounding 0.20 | brevity 0.15", "#e8f7ee"),
        ],
        3: [
            ("Optimizer: OpenAI / gpt-5.5", "#eef4ff"),
            ("Target: Anthropic / Claude Sonnet", "#fff4e6"),
            ("Credentials: saved separately, never displayed", "#e8f7ee"),
        ],
        4: [
            ("MLflow tracking URI + token", "#eef4ff"),
            ("LangSmith endpoint + project + key", "#fff4e6"),
            ("Skip tracking and run locally", "#e8f7ee"),
        ],
        5: [
            ("Candidate for approval score 0.91", "#e8f7ee"),
            ("Attempt not used score 0.88", "#fff4e6"),
            ("Cost, tokens, policy hash, data hash captured", "#eef4ff"),
        ],
        6: [
            ("Version progress: v1 0.78 -> v2 0.91", "#eef4ff"),
            ("Inspect attempts not used", "#fff4e6"),
            ("Approve best version for promotion", "#e8f7ee"),
        ],
        7: [
            ("Create Git PR", "#eef4ff"),
            ("Attach source snapshots, eval report, and lineage", "#e8f7ee"),
            ("Ready for production review", "#e8f7ee"),
        ],
        8: [
            ("Approved artifact: support-agent run_123", "#eef4ff"),
            ("Model A gpt-5.5 | Model B gpt-4o-mini", "#fff4e6"),
            ("Estimated or live eval comparison", "#e8f7ee"),
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
