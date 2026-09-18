"""Render the top of today's front pages as 800x480 1-bit PNGs for a TRMNL OG.

Each paper becomes its own image (out/<key>.png), shown on the device through
TRMNL's Alias plugin. Run: python frontpage.py [--out out] [--only nyt,ft]
"""

import argparse
import datetime as dt
import io
import json
import ssl
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SCREEN_W, SCREEN_H = 800, 480
BAR_H = 32
PAGE_W = 600  # page is shrunk to this width so more of it fits above the bar

CDN = "https://cdn.thepaperboy.com/frontpages/{region}/{ymd}/{slug}_lg.jpg"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128 Safari/537.36"

# trim_top: fraction of page width to cut from the top (printer's marks etc.)
PAPERS = [
    {"key": "nyt", "name": "The New York Times", "region": "us", "slug": "new_york_times", "trim_top": 0.075},
    {"key": "wsj", "name": "The Wall Street Journal", "region": "us", "slug": "wall_street_journal", "trim_top": 0.02},
    {"key": "guardian", "name": "The Guardian", "region": "uk", "slug": "the_guardian", "trim_top": 0.0},
    {"key": "ft", "name": "Financial Times", "region": "uk", "slug": "financial_times", "trim_top": 0.0},
]

try:
    import certifi

    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()

FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def fetch(url):
    """Return the response bytes, or None on any HTTP/network failure."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as r:
            data = r.read()
            # The CDN serves application/octet-stream, so check the JPEG signature.
            if r.status == 200 and data[:3] == b"\xff\xd8\xff":
                return data
    except Exception:
        pass
    return None


def candidate_dates(today):
    # Tomorrow first: UK papers post the next day's edition the evening before.
    return [today + dt.timedelta(days=d) for d in (1, 0, -1)]


def find_edition(paper, today, fetcher=fetch):
    """Return (edition_date, image_bytes) for the newest available edition."""
    for day in candidate_dates(today):
        url = CDN.format(region=paper["region"], ymd=day.strftime("%Y%m%d"), slug=paper["slug"])
        data = fetcher(url)
        if data:
            return day, data
    return None, None


def atkinson(gray):
    """Atkinson dither: keeps small type crisp and photos light."""
    a = np.asarray(gray, dtype=np.float32).copy()
    h, w = a.shape
    for y in range(h):
        row = a[y]
        for x in range(w):
            old = row[x]
            new = 255.0 if old >= 128 else 0.0
            row[x] = new
            err = (old - new) / 8
            if x + 1 < w:
                row[x + 1] += err
            if x + 2 < w:
                row[x + 2] += err
            if y + 1 < h:
                nxt = a[y + 1]
                if x > 0:
                    nxt[x - 1] += err
                nxt[x] += err
                if x + 1 < w:
                    nxt[x + 1] += err
            if y + 2 < h:
                a[y + 2, x] += err
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def prepare_page(img, trim_top):
    """Grayscale, scale to PAGE_W, trim the top, crop to the space above the bar."""
    g = img.convert("L")
    h = round(g.height * PAGE_W / g.width)
    g = g.resize((PAGE_W, h), Image.LANCZOS)
    top = round(PAGE_W * trim_top)
    g = g.crop((0, top, PAGE_W, top + SCREEN_H - BAR_H))
    # Levels: map the paper's own background tone (e.g. FT salmon) to white.
    a = np.asarray(g, dtype=np.float32)
    black, white = np.percentile(a, 1), np.percentile(a, 60)
    if white - black < 40:
        black, white = 0.0, 255.0
    a = ((a - black) * 255.0 / (white - black)).clip(0, 255)
    g = Image.fromarray(a.astype(np.uint8))
    return g.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=2))


def load_font(size):
    for p in FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def render(img, paper, edition, position, total):
    page = atkinson(prepare_page(img, paper["trim_top"]))
    screen = Image.new("L", (SCREEN_W, SCREEN_H), 255)
    screen.paste(page, ((SCREEN_W - PAGE_W) // 2, 0))
    d = ImageDraw.Draw(screen)
    d.rectangle((0, SCREEN_H - BAR_H, SCREEN_W, SCREEN_H), fill=0)
    font = load_font(17)
    mid = SCREEN_H - BAR_H // 2
    d.text((16, mid), paper["name"], fill=255, font=font, anchor="lm")
    label = f"{edition.strftime('%a %-d %b')}   {position}/{total}"
    d.text((SCREEN_W - 16, mid), label, fill=255, font=font, anchor="rm")
    return screen.point(lambda v: 255 if v >= 128 else 0).convert("1", dither=Image.NONE)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--only", help="comma-separated paper keys")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    only = set(args.only.split(",")) if args.only else None
    today = dt.datetime.now(dt.timezone.utc).date()

    failures = 0
    for i, paper in enumerate(PAPERS, 1):
        if only and paper["key"] not in only:
            continue
        edition, data = find_edition(paper, today)
        if not data:
            # Leave the previous image in place; its bar still shows its own date.
            print(f"{paper['key']}: no edition found, keeping {state.get(paper['key'], 'nothing')}")
            failures += 1
            continue
        if state.get(paper["key"]) == edition.isoformat() and (out / f"{paper['key']}.png").exists():
            print(f"{paper['key']}: {edition} already rendered")
            continue
        render(Image.open(io.BytesIO(data)), paper, edition, i, len(PAPERS)).save(
            out / f"{paper['key']}.png", optimize=True
        )
        state[paper["key"]] = edition.isoformat()
        print(f"{paper['key']}: rendered {edition}")

    state_path.write_text(json.dumps(state, indent=2) + "\n")
    return 1 if failures == len(PAPERS) else 0


if __name__ == "__main__":
    sys.exit(main())
