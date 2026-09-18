# render

Builds five 800×480 2-bit grayscale PNGs for a TRMNL OG and force-pushes them to the
`images` branch four times a day. Each is shown on the device with a TRMNL
**Alias** plugin (cache: off) pointing at:

    https://raw.githubusercontent.com/<owner>/<repo>/images/nyt.png
    https://raw.githubusercontent.com/<owner>/<repo>/images/wsj.png
    https://raw.githubusercontent.com/<owner>/<repo>/images/guardian.png
    https://raw.githubusercontent.com/<owner>/<repo>/images/ft.png
    https://raw.githubusercontent.com/<owner>/<repo>/images/globe.png

Local run: `pip install -r requirements.txt && python frontpage.py --out out`
Tests: `python -m unittest discover -s tests -t .`
