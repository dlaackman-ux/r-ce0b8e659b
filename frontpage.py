"""Render the top of today's front pages as 800x480 PNGs for a TRMNL OG.

Each paper becomes its own image (out/<key>.png), shown on the device through
TRMNL's Alias plugin. Run: python frontpage.py [--out out] [--only nyt,ft]
"""

import argparse
import datetime as dt
import email.utils
import io
import json
import ssl
import struct
import sys
import urllib.request
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCREEN_W, SCREEN_H = 800, 480
BAR_H = 32
PAGE_W = 600  # page is shrunk to this width so more of it fits above the bar
PAGE_H = SCREEN_H - BAR_H

# 4 = 2-bit grayscale (black, dark gray, light gray, white); 2 = pure black and white.
GRAY_LEVELS = 2
# PNG bit depth. The OG uses fast (partial) refresh for 1-bit images, which leaves
# text washed out and ghosted; a 2-bit file gets a full refresh even when it only
# contains black and white.
PNG_BITS = 2
# Bump when the rendering changes so already-published editions get redrawn.
RENDER_VERSION = 4

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128 Safari/537.36"
PAPERBOY = "https://cdn.thepaperboy.com/frontpages/{region}/{ymd}/{slug}_lg.jpg"
# Freedom Forum keeps one file per day-of-month, overwritten monthly; the PDF has real text.
FREEDOM_FORUM = "https://cdn.freedomforum.org/dfp/pdf{day}/{code}.pdf"

# trim_top: fraction of page width to cut from the top (printer's marks etc.)
PAPERS = [
    {"key": "nyt", "name": "The New York Times", "region": "us", "slug": "new_york_times", "ff": "NY_NYT", "trim_top": 0.075},
    {"key": "wsj", "name": "The Wall Street Journal", "region": "us", "slug": "wall_street_journal", "ff": "WSJ", "trim_top": 0.02},
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
    """Return (bytes, last_modified) for a JPEG or PDF, or (None, None) on any failure."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as r:
            data = r.read()
            # CDNs here serve application/octet-stream, so check the file signature.
            if r.status == 200 and (data[:3] == b"\xff\xd8\xff" or data[:5] == b"%PDF-"):
                lm = r.headers.get("Last-Modified")
                return data, email.utils.parsedate_to_datetime(lm).date() if lm else None
    except Exception:
        pass
    return None, None


def candidate_dates(today):
    # Tomorrow first: UK papers post the next day's edition the evening before.
    return [today + dt.timedelta(days=d) for d in (1, 0, -1)]


def sources_for(paper, day):
    """URLs to try for one edition date, best quality first."""
    if paper.get("ff"):
        yield FREEDOM_FORUM.format(day=day.day, code=paper["ff"]), True
    yield PAPERBOY.format(region=paper["region"], ymd=day.strftime("%Y%m%d"), slug=paper["slug"]), False


def find_edition(paper, today, fetcher=fetch):
    """Return (edition_date, file_bytes) for the newest available edition."""
    for day in candidate_dates(today):
        for url, dated_by_header in sources_for(paper, day):
            data, modified = fetcher(url)
            if not data:
                continue
            # Freedom Forum URLs only carry the day of the month; reject last month's file.
            if dated_by_header and (modified is None or abs((modified - day).days) > 1):
                continue
            return day, data
    return None, None


def page_image(data):
    """Decode a front page (PDF or JPEG) to grayscale, PAGE_W pixels wide."""
    if data[:5] == b"%PDF-":
        import pymupdf

        page = pymupdf.open(stream=data, filetype="pdf")[0]
        zoom = PAGE_W * 2 / page.rect.width  # render at 2x, then downsample
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY)
        img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    else:
        img = Image.open(io.BytesIO(data)).convert("L")
    return img.resize((PAGE_W, round(img.height * PAGE_W / img.width)), Image.LANCZOS)


def levels(img):
    """Map the paper's own background tone (e.g. FT salmon) to white and stretch contrast."""
    a = np.asarray(img, dtype=np.float32)
    black, white = np.percentile(a, 1), np.percentile(a, 60)
    if white - black < 40:
        black, white = 0.0, 255.0
    return ((a - black) * 255.0 / (white - black)).clip(0, 255)


def box_mean(m, r):
    """Mean over a (2r+1)x(2r+1) window, via an integral image."""
    k = 2 * r + 1
    p = np.pad(m, ((r + 1, r), (r + 1, r)), mode="edge").cumsum(0).cumsum(1)
    return (p[k:, k:] - p[:-k, k:] - p[k:, :-k] + p[:-k, :-k]) / (k * k)


def photo_mask(a):
    """True where the page is a photo (lots of mid-tones), False for text and rules."""
    mid = ((a > 50) & (a < 205)).astype(np.float32)
    photo = box_mean(mid, 6) > 0.35
    return box_mean(photo.astype(np.float32), 4) > 0.2  # soften the seam at photo edges


def atkinson(a, n_levels):
    """Atkinson error diffusion to n evenly spaced gray levels; keeps photos light."""
    a = a.copy()
    h, w = a.shape
    step = 255.0 / (n_levels - 1)
    for y in range(h):
        row = a[y]
        for x in range(w):
            old = row[x]
            new = min(255.0, max(0.0, round(old / step) * step))
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
    return a


def quantize(a, n_levels):
    """Text: snap to the nearest level with no dithering, so type stays solid."""
    if n_levels == 2:
        return np.where(a < 185, 0.0, 255.0)  # high cutoff keeps small type from thinning
    step = 255.0 / (n_levels - 1)
    return np.round(a / step) * step


def convert(img, n_levels):
    a = levels(img)
    out = np.where(photo_mask(a), atkinson(a, n_levels), quantize(a, n_levels))
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def load_font(size):
    for p in FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def render(data, paper, edition, position, total, n_levels=GRAY_LEVELS):
    page = page_image(data)
    top = round(PAGE_W * paper["trim_top"])
    page = page.crop((0, top, PAGE_W, top + PAGE_H))
    screen = Image.new("L", (SCREEN_W, SCREEN_H), 255)
    screen.paste(convert(page, n_levels), ((SCREEN_W - PAGE_W) // 2, 0))
    d = ImageDraw.Draw(screen)
    d.rectangle((0, PAGE_H, SCREEN_W, SCREEN_H), fill=0)
    font = load_font(17)
    mid = SCREEN_H - BAR_H // 2
    d.text((16, mid), paper["name"], fill=255, font=font, anchor="lm")
    label = f"{edition.strftime('%a %-d %b')}   {position}/{total}"
    d.text((SCREEN_W - 16, mid), label, fill=255, font=font, anchor="rm")
    return Image.fromarray(quantize(np.asarray(screen, dtype=np.float32), n_levels).astype(np.uint8))


def png_bytes(img, bits=PNG_BITS):
    """Encode as a true 1- or 2-bit grayscale PNG (Pillow only writes 8-bit grayscale)."""
    max_idx = (1 << bits) - 1
    idx = (np.asarray(img, dtype=np.uint16) * max_idx + 127) // 255  # 0..max_idx
    h, w = idx.shape
    per_byte = 8 // bits
    padded = np.zeros((h, -(-w // per_byte) * per_byte), dtype=np.uint8)
    padded[:, :w] = idx
    groups = padded.reshape(h, -1, per_byte)
    packed = np.zeros(groups.shape[:2], dtype=np.uint8)
    for i in range(per_byte):
        packed |= groups[:, :, i] << (8 - bits * (i + 1))
    raw = b"".join(b"\x00" + row.tobytes() for row in packed)

    def chunk(tag, body):
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, bits, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


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
        key = paper["key"]
        if only and key not in only:
            continue
        edition, data = find_edition(paper, today)
        if not data:
            # Leave the previous image in place; its bar still shows its own date.
            print(f"{key}: no edition found, keeping {state.get(key, 'nothing')}")
            failures += 1
            continue
        stamp = f"{edition.isoformat()} v{RENDER_VERSION}"
        if state.get(key) == stamp and (out / f"{key}.png").exists():
            print(f"{key}: {edition} already rendered")
            continue
        (out / f"{key}.png").write_bytes(png_bytes(render(data, paper, edition, i, len(PAPERS))))
        state[key] = stamp
        print(f"{key}: rendered {edition} ({'pdf' if data[:5] == b'%PDF-' else 'jpg'})")

    state_path.write_text(json.dumps(state, indent=2) + "\n")
    return 1 if failures == len(PAPERS) else 0


if __name__ == "__main__":
    sys.exit(main())
