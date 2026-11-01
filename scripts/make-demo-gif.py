#!/usr/bin/env python3
"""Build examples/demo.gif + examples/demo-transcript.txt from REAL servefit runs.

Every command's output is captured verbatim from the built CLI
(`python -m servefit`) running against synthetic inventories. Nothing is
invented or reworded: the script only adds a typing animation and hold frames.
The probe step runs against a scripted local fake OpenAI-compatible server
(same fixture style as tests/test_probe.py), labeled as such in the transcript.
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_GIF = os.path.join(ROOT, "examples", "demo.gif")
OUT_TXT = os.path.join(ROOT, "examples", "demo-transcript.txt")
PROFILE = os.path.join(ROOT, "examples", "profile-rtx4090.json")

BG = (30, 30, 30)
FG = (212, 212, 212)
PROMPT = (126, 231, 135)
CMD = (255, 255, 255)

FPS = 12
FONT_SIZE = 18
LINE_H = 24
PAD = 16


class FakeHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        time.sleep(0.25)
        for word in ["Fifteen", "tokens", "per", "second", "on", "this", "box"]:
            payload = json.dumps({"choices": [{"delta": {"content": word + " "}}]})
            self.wfile.write(("data: %s\n\n" % payload).encode())
            time.sleep(0.05)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, *a):
        pass


def sh(cmd, env):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    return r.returncode, r.stdout, r.stderr


def main():
    sys.path.insert(0, ROOT)
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["COLUMNS"] = "100"

    srv = HTTPServer(("127.0.0.1", 0), FakeHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]

    script = [
        (sys.executable + " -m servefit --version", 10),
        (sys.executable + " -m servefit inventory --from examples/profile-rtx4090.json", 40),
        (sys.executable + " -m servefit fit --model Qwen3-8B --from examples/profile-rtx4090.json "
         "--no-hf --params 8 --dtype bf16 --layers 36 --kv-heads 8 --head-dim 128 "
         "--context 32768", 90),
        (sys.executable + " -m servefit probe --endpoint http://127.0.0.1:%d --model demo-8b" % port, 50),
    ]

    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", FONT_SIZE)
    probe = Image.new("RGB", (8, 8))
    d0 = ImageDraw.Draw(probe)
    char_w = d0.textlength("0123456789" * 10, font=font) / 100

    steps = []
    for cmd, hold in script:
        code, out, err = sh(cmd.split(), env)
        text = out + err
        steps.append((cmd, code, text, hold))

    with open(OUT_TXT, "w") as f:
        f.write("# demo transcript: real servefit output; probe ran against a\n"
                "# scripted local fake OpenAI-compatible server (timings scripted).\n\n")
        for cmd, code, text, _ in steps:
            f.write("$ " + cmd + "\n")
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
            if code != 0:
                f.write("(exit code %d)\n" % code)

    lines = []
    for cmd, _, text, _ in steps:
        lines.append("$ " + cmd)
        for ln in text.split("\n"):
            if ln == "" and text.endswith("\n"):
                continue
            lines.append(ln)
    max_len = max(len(t) for t in lines)
    width = max(960, int(max_len * char_w) + PAD * 2)
    visible = 26
    height = visible * LINE_H + PAD * 2

    frames = []
    cur = []
    for cmd, _, text, hold in steps:
        for n in range(1, len(cmd) + 1, 3):
            frames.append(draw_frame(cur, "$ " + cmd[:n], width, height, font, char_w, visible))
        cur.append("$ " + cmd)
        for _ in range(6):
            frames.append(draw_frame(cur, None, width, height, font, char_w, visible))
        for ln in text.split("\n"):
            if ln == "" and text.endswith("\n"):
                continue
            cur.append(ln)
        frames.append(draw_frame(cur, None, width, height, font, char_w, visible))
        for _ in range(hold):
            frames.append(draw_frame(cur, None, width, height, font, char_w, visible))
    for _ in range(12):
        frames.append(draw_frame(cur, None, width, height, font, char_w, visible))

    frames[0].save(OUT_GIF, save_all=True, append_images=frames[1:],
                   duration=int(1000 / FPS), loop=0)
    print("wrote %s (%d frames, %d bytes)" % (OUT_GIF, len(frames), os.path.getsize(OUT_GIF)))
    print("wrote %s" % OUT_TXT)
    srv.shutdown()


def draw_frame(done_lines, typing, width, height, font, char_w, visible):
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)
    items = list(done_lines)
    if typing is not None:
        items.append(typing)
    start = max(0, len(items) - visible)
    y = PAD
    for text in items[start:]:
        if text.startswith("$ "):
            d.text((PAD, y), "$ ", font=font, fill=PROMPT)
            d.text((PAD + 2 * char_w, y), text[2:], font=font, fill=CMD)
        else:
            d.text((PAD, y), text, font=font, fill=FG)
        y += LINE_H
    return img


if __name__ == "__main__":
    main()
