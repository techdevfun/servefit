"""Tests for servefit.probe against a scripted fake OpenAI-compatible server."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from servefit import probe as probemod

FIRST_CHUNK_DELAY = 0.25
INTER_CHUNK_DELAY = 0.05
CHUNKS = ["Hello", "world", "this", "is", "a", "test"]


class FakeHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        assert self.path == "/v1/chat/completions"
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def emit(payload):
            self.wfile.write(("data: %s\n\n" % json.dumps(payload)).encode())

        time.sleep(FIRST_CHUNK_DELAY)
        for i, word in enumerate(CHUNKS):
            emit({"choices": [{"delta": {"content": word + " "},
                               "finish_reason": None if i < len(CHUNKS) - 1 else "stop"}]})
            time.sleep(INTER_CHUNK_DELAY)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, *a):
        pass


def _server():
    srv = HTTPServer(("127.0.0.1", 0), FakeHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


def test_probe_measures_scripted_timings():
    srv = _server()
    try:
        url = "http://127.0.0.1:%d" % srv.server_address[1]
        res = probemod.probe_endpoint(url, "fake-model", "hi", max_tokens=16)
    finally:
        srv.shutdown()
    assert res["ok"] is True
    assert res["stream_done"] is True
    # TTFT should be ~FIRST_CHUNK_DELAY (allow generous tolerance for CI jitter)
    assert abs(res["ttft_s"] - FIRST_CHUNK_DELAY) < 0.4, res["ttft_s"]
    assert res["approx_tokens"] == len(CHUNKS)
    assert res["tokens_per_s"] > 0
    assert res["caveats"], "probe must label its numbers"


def test_probe_reports_connection_error():
    res = probemod.probe_endpoint("http://127.0.0.1:1", "x", "hi", timeout=2)
    assert res["ok"] is False
    assert "error" in res
