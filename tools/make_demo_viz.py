"""Demo gallery: per clip a video of the six cameras laid out as on the car (the front input in the middle of the top row,
the generated cross views beside it, the generated rear views below), next to the ground truth when available, and an
index.html that plays them.

    python tools/make_demo_viz.py --renders RENDERS --out GALLERY                      # generated views only
    python tools/make_demo_viz.py --renders RENDERS --gt GT --out GALLERY --with-gt    # + the ground truth beside them

RENDERS holds the inference outputs (front_input.mp4 and pred_<view>.mp4 per clip). The ground truth is read from
gt_<view>.mp4 next to the predictions (cached inference writes them), else from GT/<clip>/gt_<view>.mp4.
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from openlongtail.inference.video import write_mp4

GRID = (('cross_left', 'front', 'cross_right'), ('rear_left', 'rear_tele', 'rear_right'))     # as seen from above the car
TITLES = {'front': 'FRONT', 'cross_left': 'CROSS LEFT', 'cross_right': 'CROSS RIGHT', 'rear_left': 'REAR LEFT',
          'rear_tele': 'REAR TELE', 'rear_right': 'REAR RIGHT'}
TILE_W, TILE_H, GAP, PAD, MARGIN = 416, 240, 6, 14, 24
PANEL_HEADER, HEADER, FOOTER = 44, 64, 36
FPS, NUM_FRAMES = 10, 41
BACKGROUND, PANEL, TEXT, DIM_TEXT = (15, 17, 21), (24, 27, 34), (236, 238, 243), (150, 156, 170)
INPUT_COLOR, PRED_COLOR, GT_COLOR = (79, 163, 255), (245, 185, 66), (110, 214, 150)
FONT_PATHS = {False: ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/System/Library/Fonts/Supplemental/Arial.ttf',
                      'C:/Windows/Fonts/arial.ttf'),
              True: ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
                     'C:/Windows/Fonts/arialbd.ttf')}


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for path in FONT_PATHS[bold]:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default(size)


FONT_TITLE, FONT_PANEL, FONT_LABEL, FONT_SMALL = load_font(22, True), load_font(17, True), load_font(13, True), load_font(14)


def load_video(path: Path) -> np.ndarray | None:
    return iio.imread(path, plugin='pyav')[:NUM_FRAMES] if path.exists() else None


def fit(frame: np.ndarray) -> Image.Image:
    """Letterbox a frame into a tile."""
    img = Image.fromarray(frame).convert('RGB')
    img.thumbnail((TILE_W, TILE_H), Image.BILINEAR)
    tile = Image.new('RGB', (TILE_W, TILE_H), (0, 0, 0))
    tile.paste(img, ((TILE_W - img.width) // 2, (TILE_H - img.height) // 2))
    return tile


def label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, color: tuple[int, int, int]) -> None:
    """A small pill with the camera name, in the corner of a tile."""
    w = int(draw.textlength(text, font=FONT_LABEL)) + 30
    draw.rounded_rectangle([x + 8, y + 8, x + 8 + w, y + 30], radius=11, fill=(10, 11, 14))
    draw.ellipse([x + 15, y + 15, x + 23, y + 23], fill=color)
    draw.text((x + 28, y + 11), text, font=FONT_LABEL, fill=TEXT)


def panel(canvas: Image.Image, x0: int, y0: int, videos: dict, prefix: str, t: int, title: str, color: tuple[int, int, int]) -> None:
    draw = ImageDraw.Draw(canvas)
    width = 3 * TILE_W + 2 * GAP + 2 * PAD
    height = PANEL_HEADER + 2 * TILE_H + GAP + PAD
    draw.rounded_rectangle([x0, y0, x0 + width, y0 + height], radius=12, fill=PANEL)
    draw.rectangle([x0 + PAD, y0 + 16, x0 + PAD + 4, y0 + 34], fill=color)
    draw.text((x0 + PAD + 14, y0 + 14), title, font=FONT_PANEL, fill=TEXT)
    for r, row in enumerate(GRID):
        for c, view in enumerate(row):
            x, y = x0 + PAD + c * (TILE_W + GAP), y0 + PANEL_HEADER + r * (TILE_H + GAP)
            video = videos.get('front' if view == 'front' else f'{prefix}_{view}')
            if video is None or len(video) == 0:
                canvas.paste(Image.new('RGB', (TILE_W, TILE_H), (40, 34, 36)), (x, y))
                draw.text((x + TILE_W // 2 - 30, y + TILE_H // 2 - 8), 'no data', font=FONT_SMALL, fill=DIM_TEXT)
            else:
                canvas.paste(fit(video[min(t, len(video) - 1)]), (x, y))
            tile_color = INPUT_COLOR if view == 'front' else color
            draw.rectangle([x, y, x + TILE_W - 1, y + TILE_H - 1], outline=tile_color, width=2 if view == 'front' else 1)
            label(draw, x, y, TITLES[view] + ('  ·  INPUT' if view == 'front' else ''), tile_color)


def canvas_size(show_gt: bool) -> tuple[int, int]:
    panel_w = 3 * TILE_W + 2 * GAP + 2 * PAD
    W = 2 * MARGIN + panel_w + (panel_w + 16 if show_gt else 0)
    H = HEADER + PANEL_HEADER + 2 * TILE_H + GAP + PAD + FOOTER
    return W + (-W) % 16, H + (-H) % 16                      # multiples of 16 for the video encoder


def compose(videos: dict[str, np.ndarray | None], t: int, name: str, show_gt: bool) -> np.ndarray:
    W, H = canvas_size(show_gt)
    canvas = Image.new('RGB', (W, H), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((MARGIN, 16), 'OpenLongTail', font=FONT_TITLE, fill=TEXT)
    draw.text((MARGIN + int(draw.textlength('OpenLongTail', font=FONT_TITLE)) + 14, 22),
              'five surround cameras generated from a single front-camera video', font=FONT_SMALL, fill=DIM_TEXT)
    stamp = f'{name}    frame {t + 1:02d} / {NUM_FRAMES}'
    draw.text((W - MARGIN - int(draw.textlength(stamp, font=FONT_SMALL)), 22), stamp, font=FONT_SMALL, fill=DIM_TEXT)
    panel(canvas, MARGIN, HEADER, videos, 'pred', t, 'Generated', PRED_COLOR)
    if show_gt:
        panel(canvas, MARGIN + 3 * TILE_W + 2 * GAP + 2 * PAD + 16, HEADER, videos, 'gt', t, 'Ground truth', GT_COLOR)
    legend = [(INPUT_COLOR, 'real input (front camera)'), (PRED_COLOR, 'generated by OpenLongTail')]
    legend += [(GT_COLOR, 'ground truth')] if show_gt else []
    x = MARGIN
    for color, text in legend:
        draw.ellipse([x, H - 25, x + 10, H - 15], fill=color)
        draw.text((x + 16, H - 29), text, font=FONT_SMALL, fill=DIM_TEXT)
        x += 40 + int(draw.textlength(text, font=FONT_SMALL))
    return np.asarray(canvas)


def find_clips(renders: Path) -> dict[str, Path]:
    """Clip name -> directory of every clip with predictions (the scene uuid when the directory is clip_<id>)."""
    clips = {}
    for pred in renders.rglob('pred_cross_left.mp4'):
        clip_dir = pred.parent
        clips[clip_dir.parent.name if clip_dir.name.startswith('clip_') else clip_dir.name] = clip_dir
    return dict(sorted(clips.items()))


def write_index(out: Path, cards: list[tuple[str, str]], show_gt: bool) -> None:
    note = ('Only the front camera (blue) is real; the other five cameras are generated (amber)'
            + (', shown beside the ground truth (green).' if show_gt else '.'))
    items = '\n'.join(f'<figure><video src="{stem}.mp4" poster="{stem}.png" autoplay muted loop playsinline controls></video>'
                      f'<figcaption>{html.escape(name)}</figcaption></figure>' for stem, name in cards)
    (out / 'index.html').write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenLongTail demo</title>
<style>
:root{{color-scheme:dark}}
body{{margin:0;background:#0f1115;color:#eceef3;font:15px/1.5 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}}
main{{max-width:1680px;margin:0 auto;padding:32px 24px 56px}}
h1{{margin:0 0 6px;font-size:26px;font-weight:650;letter-spacing:-.01em}}
p{{margin:0 0 28px;color:#969caa;max-width:900px}}
figure{{margin:0 0 28px;background:#181b22;border:1px solid #262a34;border-radius:14px;overflow:hidden}}
video{{display:block;width:100%;height:auto;background:#000}}
figcaption{{padding:10px 16px;color:#969caa;font:13px ui-monospace,Menlo,monospace}}
</style></head><body><main>
<h1>OpenLongTail demo</h1>
<p>Five surround cameras generated from a single front-camera video, laid out as on the car. {note}</p>
{items}
</main></body></html>
""")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--renders', type=Path, required=True)
    ap.add_argument('--gt', type=Path, default=None, help='directory of <clip>/gt_<view>.mp4 and front_input.mp4')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--with-gt', action='store_true')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    clips = find_clips(a.renders)
    if not clips:
        raise SystemExit(f'no pred_*.mp4 under {a.renders}')
    cards = []
    for k, (name, clip_dir) in enumerate(clips.items(), 1):
        views = [v for row in GRID for v in row if v != 'front']
        videos = {'front': load_video(clip_dir / 'front_input.mp4')}
        videos |= {f'pred_{v}': load_video(clip_dir / f'pred_{v}.mp4') for v in views}
        show_gt = a.with_gt
        if show_gt:
            gt_dir = a.gt / name if a.gt is not None else None
            for v in views:
                gt = load_video(clip_dir / f'gt_{v}.mp4')
                videos[f'gt_{v}'] = gt if gt is not None or gt_dir is None else load_video(gt_dir / f'gt_{v}.mp4')
            show_gt = any(videos[f'gt_{v}'] is not None for v in views)
        length = min(max([len(v) for v in videos.values() if v is not None] + [1]), NUM_FRAMES)
        frames = [compose(videos, t, name, show_gt) for t in range(length)]
        stem = f'{k:02d}_{name[:16]}'
        Image.fromarray(frames[length // 2]).save(a.out / f'{stem}.png')
        write_mp4(a.out / f'{stem}.mp4', np.stack(frames), FPS)
        cards.append((stem, name))
        print(f'[{k}/{len(clips)}] {name}', flush=True)
    write_index(a.out, cards, a.with_gt)
    print(f'wrote {a.out / "index.html"} ({len(cards)} clips)', flush=True)


if __name__ == '__main__':
    main()
