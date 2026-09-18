import datetime as dt
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

import frontpage as fp

TODAY = dt.date(2026, 9, 18)


def fake_page(w=700, h=1340, tint=255):
    img = Image.new("RGB", (w, h), (tint, tint, tint))
    img.paste((0, 0, 0), (50, 80, 650, 160))  # a "masthead"
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return buf.getvalue()


class FindEdition(unittest.TestCase):
    def test_prefers_tomorrow_then_today(self):
        seen = []

        def fetcher(url):
            seen.append(url)
            return b"img" if "20260918" in url else None

        day, data = fp.find_edition(fp.PAPERS[0], TODAY, fetcher)
        self.assertEqual(day, TODAY)
        self.assertEqual(data, b"img")
        self.assertIn("20260919", seen[0])
        self.assertTrue(seen[1].endswith("/us/20260918/new_york_times_lg.jpg"))

    def test_nothing_available(self):
        self.assertEqual(fp.find_edition(fp.PAPERS[2], TODAY, lambda u: None), (None, None))


class Render(unittest.TestCase):
    def test_output_is_800x480_one_bit(self):
        img = Image.open(io.BytesIO(fake_page()))
        out = fp.render(img, fp.PAPERS[3], TODAY, 4, 4)
        self.assertEqual(out.size, (800, 480))
        self.assertEqual(out.mode, "1")
        # Bottom bar is mostly black, side margins are white.
        bar = out.crop((0, 450, 800, 480)).convert("L")
        self.assertLess(sum(bar.getdata()) / (800 * 30), 60)
        self.assertEqual(out.getpixel((10, 200)), 255)

    def test_tinted_paper_background_becomes_white(self):
        img = Image.open(io.BytesIO(fake_page(tint=235)))  # FT-style salmon, as gray
        page = fp.prepare_page(img, 0.0)
        self.assertGreater(page.getpixel((300, 400)), 240)


class Main(unittest.TestCase):
    def test_keeps_old_image_when_source_missing(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "nyt.png").write_bytes(b"old")
            (out / "state.json").write_text(json.dumps({"nyt": "2026-09-17"}))

            def find(paper, today):
                return (None, None) if paper["key"] == "nyt" else (TODAY, fake_page())

            with mock.patch.object(fp, "find_edition", find):
                rc = fp.main(["--out", d])
            self.assertEqual(rc, 0)
            self.assertEqual((out / "nyt.png").read_bytes(), b"old")
            state = json.loads((out / "state.json").read_text())
            self.assertEqual(state["nyt"], "2026-09-17")
            self.assertEqual(state["ft"], "2026-09-18")
            self.assertEqual(Image.open(out / "ft.png").size, (800, 480))

    def test_all_sources_missing_fails(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(fp, "find_edition", lambda p, t: (None, None)):
                self.assertEqual(fp.main(["--out", d]), 1)


if __name__ == "__main__":
    unittest.main()
