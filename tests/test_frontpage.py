import datetime as dt
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw

import frontpage as fp

TODAY = dt.date(2026, 9, 18)


def fake_page(w=700, h=1340, tint=255):
    img = Image.new("RGB", (w, h), (tint, tint, tint))
    img.paste((0, 0, 0), (50, 80, 650, 160))  # a "masthead"
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return buf.getvalue()


class FindEdition(unittest.TestCase):
    def test_prefers_tomorrow_then_pdf_then_jpg(self):
        seen = []

        def fetcher(url):
            seen.append(url)
            return (b"jpg", None) if url.endswith("/us/20260918/new_york_times_lg.jpg") else (None, None)

        day, data = fp.find_edition(fp.PAPERS[0], TODAY, fetcher)
        self.assertEqual((day, data), (TODAY, b"jpg"))
        self.assertIn("pdf19/NY_NYT.pdf", seen[0])
        self.assertIn("pdf18/NY_NYT.pdf", seen[2])

    def test_rejects_last_months_freedom_forum_file(self):
        def fetcher(url):
            if "freedomforum" in url:
                return b"%PDF-old", dt.date(2026, 8, 18)
            return (b"jpg", None) if "20260918" in url else (None, None)

        self.assertEqual(fp.find_edition(fp.PAPERS[1], TODAY, fetcher), (TODAY, b"jpg"))

    def test_accepts_current_freedom_forum_file(self):
        def fetcher(url):
            return (b"%PDF-new", TODAY) if "pdf18" in url else (None, None)

        self.assertEqual(fp.find_edition(fp.PAPERS[0], TODAY, fetcher), (TODAY, b"%PDF-new"))

    def test_uk_papers_skip_freedom_forum(self):
        seen = []
        fp.find_edition(fp.PAPERS[3], TODAY, lambda u: (seen.append(u), (None, None))[1])
        self.assertTrue(all("thepaperboy" in u for u in seen))


    def test_freedom_forum_only_paper(self):
        seen = []
        globe = next(p for p in fp.PAPERS if p["key"] == "globe")
        fp.find_edition(globe, TODAY, lambda u: (seen.append(u), (None, None))[1])
        self.assertTrue(seen)
        self.assertTrue(all("CAN_TGAM.pdf" in u for u in seen))


class Render(unittest.TestCase):
    def test_output_uses_only_allowed_levels(self):
        out = fp.render(fake_page(), fp.PAPERS[3], TODAY, 4, 4)
        self.assertEqual(out.size, (800, 480))
        self.assertLessEqual(set(np.unique(np.asarray(out))), {0, 85, 170, 255})
        self.assertEqual(out.getpixel((10, 200)), 255)  # white side margin
        self.assertLess(np.asarray(out)[455:, :].mean(), 60)  # black bar

    def test_black_and_white_mode(self):
        out = fp.render(fake_page(), fp.PAPERS[3], TODAY, 4, 4, n_levels=2)
        self.assertLessEqual(set(np.unique(np.asarray(out))), {0, 255})

    def test_tinted_paper_background_becomes_white(self):
        page = Image.open(io.BytesIO(fake_page(tint=235))).convert("L")
        a = fp.levels(page)
        self.assertGreater(a[700, 300], 240)

    def test_text_is_not_dithered_but_photos_are(self):
        img = Image.new("L", (400, 200), 255)
        d = ImageDraw.Draw(img)
        d.text((10, 20), "Headline text here", fill=0, font=fp.load_font(28))
        grad = np.tile(np.linspace(0, 255, 150, dtype=np.uint8), (150, 1))
        img.paste(Image.fromarray(grad), (240, 40))
        a = fp.levels(img)
        mask = fp.photo_mask(a)
        self.assertFalse(mask[30:50, 10:200].any())
        self.assertTrue(mask[60:170, 280:350].all())  # mid-tone part of the gradient


class Png(unittest.TestCase):
    def test_two_bit_roundtrip(self):
        a = np.array([[0, 85, 170, 255, 255], [255, 170, 85, 0, 0]], dtype=np.uint8)
        data = fp.png_bytes(Image.fromarray(a), bits=2)
        self.assertEqual((data[24], data[25]), (2, 0))  # bit depth 2, grayscale
        back = np.asarray(Image.open(io.BytesIO(data)).convert("L"))
        np.testing.assert_array_equal(back, a)

    def test_one_bit_roundtrip(self):
        a = np.array([[0, 255] * 5 + [0]], dtype=np.uint8)
        data = fp.png_bytes(Image.fromarray(a), bits=1)
        self.assertEqual(data[24], 1)
        back = np.asarray(Image.open(io.BytesIO(data)).convert("L"))
        np.testing.assert_array_equal(back, a)


    def test_black_and_white_in_a_two_bit_file(self):
        a = np.array([[0, 255, 0, 255, 255]], dtype=np.uint8)
        data = fp.png_bytes(Image.fromarray(a), bits=2)
        self.assertEqual(data[24], 2)
        np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(data)).convert("L")), a)


class Main(unittest.TestCase):
    def test_keeps_old_image_when_source_missing(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "nyt.png").write_bytes(b"old")
            (out / "state.json").write_text(json.dumps({"nyt": "2026-09-17 v2"}))

            def find(paper, today):
                return (None, None) if paper["key"] == "nyt" else (TODAY, fake_page())

            with mock.patch.object(fp, "find_edition", find):
                rc = fp.main(["--out", d])
            self.assertEqual(rc, 0)
            self.assertEqual((out / "nyt.png").read_bytes(), b"old")
            state = json.loads((out / "state.json").read_text())
            self.assertEqual(state["nyt"], "2026-09-17 v2")
            self.assertEqual(state["ft"], f"2026-09-18 v{fp.RENDER_VERSION}")
            self.assertEqual(Image.open(out / "ft.png").size, (800, 480))

    def test_rerenders_when_render_version_changes(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "ft.png").write_bytes(b"old")
            (out / "state.json").write_text(json.dumps({"ft": "2026-09-18"}))
            with mock.patch.object(fp, "find_edition", lambda p, t: (TODAY, fake_page())):
                fp.main(["--out", d, "--only", "ft"])
            self.assertNotEqual((out / "ft.png").read_bytes(), b"old")

    def test_all_sources_missing_fails(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(fp, "find_edition", lambda p, t: (None, None)):
                self.assertEqual(fp.main(["--out", d]), 1)


if __name__ == "__main__":
    unittest.main()
