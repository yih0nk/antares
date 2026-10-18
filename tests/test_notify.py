"""Tests for the webhook alerts (``antares.notify``).

Results are built with the same exact ``simulate`` as ``tests/test_check.py``
so the scores are the offsets we dial in; ``send`` is exercised against a
real ``http.server`` on localhost.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from antares import notify
from antares.calibrate import CalibrationResult, SimParams
from antares.check import check
from antares.data import Episode
from antares.lock import RobotLock

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def _simulate(params, ep):
    return np.radians(ep.state), np.radians(ep.action)


def _lock(name="unit-a"):
    result = CalibrationResult(
        params=SimParams(), baseline=SimParams(), train_score=0.9, eval_score=1.0,
        baseline_train_score=4.0, baseline_eval_score=4.0,
        per_joint_before={j: 4.0 for j in JOINTS}, per_joint_after={j: 1.0 for j in JOINTS},
        history=[], n_evaluations=0, elapsed_s=0.0,
    )
    eps = _episodes(1, 0.5)
    return RobotLock.create("fake/robot", "so101_follower", eps, result, name=name)


def _episodes(n, offset_deg, per_joint=None):
    off = np.full(6, float(offset_deg))
    for j, v in (per_joint or {}).items():
        off[JOINTS.index(j)] = v
    eps = []
    for i in range(n):
        a = np.linspace(0, 10, 20)[:, None].repeat(6, axis=1)
        eps.append(Episode(index=i, fps=30.0, joint_names=JOINTS, state=a + off, action=a))
    return eps


def result(offset_deg, n=6, per_joint=None, **kw):
    """A CheckResult against a lock with held-out score 1.0 (limit 1.5)."""
    return check(_episodes(n, offset_deg, per_joint), _lock(), simulate=_simulate, dataset="fake/robot", **kw)


OK = result(1.2)
DRIFTED = result(2.1, per_joint={"shoulder_pan": 2.3, "shoulder_lift": 1.0, "gripper": 1.0})
NO_DATA = result(0.5, n=2)


# -- text -----------------------------------------------------------------------


def test_render_text_drifted_names_the_worst_and_failed_joints():
    text = notify.render_text(DRIFTED)
    lines = text.splitlines()
    assert lines[0] == "DRIFTED unit-a on fake/robot: 1.98 deg RMSE (limit 1.50)"
    assert lines[1] == "worst joint shoulder_pan: 2.30 deg (limit 1.50)"
    # shoulder_lift and gripper are at 1.0 (under the limit); four joints fail, three are named
    assert lines[2].startswith("failed joints (4): shoulder_pan 2.30/1.50, ")
    assert lines[2].endswith(", +1 more")
    assert "shoulder_lift" not in lines[2] and "gripper" not in lines[2]
    assert lines[3].startswith("6 episodes, 120 frames, ")
    assert lines[3].endswith("+00:00")  # ISO UTC timestamp
    assert len(lines) == 4


def test_render_text_ok_and_no_data():
    ok = notify.render_text(OK).splitlines()
    assert ok[0] == "OK unit-a on fake/robot: 1.20 deg RMSE (limit 1.50)"
    assert ok[1].startswith("worst joint ")
    assert not any(l.startswith("failed joints") for l in ok)
    assert ok[-1].startswith("6 episodes, 120 frames, ")

    nd = notify.render_text(NO_DATA).splitlines()
    assert nd[0] == "NO DATA unit-a on fake/robot: 2 episode(s), need at least 5"
    assert len(nd) == 2 and nd[1].startswith("2 episodes, 40 frames, ")


def test_render_text_title_goes_first():
    lines = notify.render_text(OK, title="Arm A nightly").splitlines()
    assert lines[0] == "Arm A nightly" and lines[1].startswith("OK unit-a")


# -- payloads -------------------------------------------------------------------


@pytest.mark.parametrize("res", [OK, DRIFTED, NO_DATA])
def test_render_payload_json_envelope(res):
    p = notify.render_payload(res, "json")
    assert set(p) == {"event", "verdict", "text", "result"}
    assert p["event"] == "antares.check" and p["verdict"] == res.verdict
    assert p["text"].splitlines()[0] == notify.headline(res)
    assert p["result"] == res.to_dict()
    json.dumps(p)


@pytest.mark.parametrize("res", [OK, DRIFTED, NO_DATA])
def test_render_payload_slack_blocks(res):
    p = notify.render_payload(res, "slack")
    assert set(p) == {"text", "blocks"}
    assert p["text"].splitlines()[0] == notify.headline(res)
    header, section, *rest = p["blocks"]
    assert header["type"] == "header" and header["text"]["type"] == "plain_text"
    assert header["text"]["text"] == f"Antares check: {notify.headline(res).split(' unit-a')[0]}"
    assert section["type"] == "section" and section["text"]["type"] == "mrkdwn"
    assert section["text"]["text"].startswith("*" + notify.headline(res) + "*")
    if res.per_joint:
        (fields,) = rest
        assert fields["type"] == "section"
        assert [f["text"].split("*")[1] for f in fields["fields"]] == list(JOINTS)
        assert all(f["type"] == "mrkdwn" for f in fields["fields"])
        if res.verdict == "drifted":
            assert any("FAIL" in f["text"] for f in fields["fields"])
        else:
            assert not any("FAIL" in f["text"] for f in fields["fields"])
    else:
        assert rest == []  # no joints were scored
    json.dumps(p)


def test_render_payload_slack_title_is_the_header():
    p = notify.render_payload(DRIFTED, "slack", title="Arm A")
    assert p["blocks"][0]["text"]["text"] == "Arm A"
    assert p["text"].startswith("Arm A\nDRIFTED unit-a")
    assert "Arm A" not in p["blocks"][1]["text"]["text"]


def test_render_payload_rejects_unknown_format():
    with pytest.raises(ValueError, match="unknown notify format"):
        notify.render_payload(OK, "auto")
    with pytest.raises(ValueError):
        notify.render_payload(OK, "teams")


def test_render_error_payload():
    j = notify.render_error_payload("lock 'ghost' not found", "json", lock="ghost", dataset="fake/robot")
    assert j["event"] == "antares.check" and j["verdict"] == "error" and j["result"] is None
    assert j["text"].startswith("ERROR ghost on fake/robot: lock 'ghost' not found")
    assert j["error"] == "lock 'ghost' not found"
    s = notify.render_error_payload("boom", "slack", lock="ghost", dataset="fake/robot")
    assert s["blocks"][0]["text"]["text"] == "Antares check: ERROR"
    assert s["blocks"][1]["text"]["text"].startswith("*ERROR ghost on fake/robot: boom*")


# -- policy ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "when, expected",
    [
        ("drifted", {"ok": False, "drifted": True, "insufficient_data": False}),
        ("failure", {"ok": False, "drifted": True, "insufficient_data": True}),
        ("always", {"ok": True, "drifted": True, "insufficient_data": True}),
    ],
)
def test_should_notify_matrix(when, expected):
    for res in (OK, DRIFTED, NO_DATA):
        assert notify.should_notify(res, when) is expected[res.verdict], (when, res.verdict)


def test_should_notify_rejects_unknown_when():
    with pytest.raises(ValueError, match="unknown notify-when"):
        notify.should_notify(OK, "sometimes")


def test_resolve_format_detects_slack_by_host():
    assert notify.resolve_format("https://hooks.slack.com/services/T0/B0/xyz") == "slack"
    assert notify.resolve_format("https://HOOKS.SLACK.COM/services/T0/B0/xyz") == "slack"
    assert notify.resolve_format("https://example.com/hooks.slack.com") == "json"
    assert notify.resolve_format("https://hooks.slack.com.evil.example/x") == "json"
    assert notify.resolve_format("https://hooks.slack.com/x", "json") == "json"
    assert notify.resolve_format("https://example.com/x", "slack") == "slack"
    with pytest.raises(ValueError):
        notify.resolve_format("https://example.com/x", "xml")


# -- transport ------------------------------------------------------------------


class _Hook(BaseHTTPRequestHandler):
    status = 200
    received: list[dict] = []

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        type(self).received.append({
            "path": self.path,
            "content_type": self.headers.get("Content-Type"),
            "user_agent": self.headers.get("User-Agent"),
            "body": json.loads(body),
        })
        self.send_response(type(self).status)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ok" if type(self).status < 400 else b"invalid_payload")

    def log_message(self, *a):  # keep the test output quiet
        pass


@pytest.fixture
def hook():
    handler = type("Hook", (_Hook,), {"status": 200, "received": []})
    srv = HTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    handler.url = f"http://127.0.0.1:{srv.server_port}/hook"
    try:
        yield handler
    finally:
        srv.shutdown()
        srv.server_close()


def test_send_posts_json_and_returns_status(hook):
    payload = notify.render_payload(DRIFTED, "json")
    assert notify.send(hook.url, payload) == 200
    (req,) = hook.received
    assert req["path"] == "/hook"
    assert req["content_type"] == "application/json"
    assert req["user_agent"] == "antares-notify"
    assert req["body"] == payload


def test_send_raises_on_5xx(hook):
    hook.status = 500
    with pytest.raises(notify.NotifyError, match="HTTP 500") as e:
        notify.send(hook.url, {"text": "x"})
    assert "invalid_payload" in str(e.value)
    assert len(hook.received) == 1


def test_send_raises_when_unreachable():
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # nothing listens here now
    with pytest.raises(notify.NotifyError, match="could not reach webhook"):
        notify.send(f"http://127.0.0.1:{port}/", {"text": "x"}, timeout=2.0)


def test_send_rejects_non_http_urls():
    with pytest.raises(notify.NotifyError, match="http"):
        notify.send("file:///etc/passwd", {"text": "x"})
    with pytest.raises(notify.NotifyError):
        notify.send("not a url", {"text": "x"})


# -- notify ---------------------------------------------------------------------


def test_notify_sends_only_when_policy_says_so(hook):
    assert notify.notify(OK, hook.url) is False
    assert hook.received == []
    assert notify.notify(DRIFTED, hook.url) is True
    assert notify.notify(NO_DATA, hook.url, when="failure") is True
    assert notify.notify(OK, hook.url, when="always", fmt="slack", title="Arm A") is True
    bodies = [r["body"] for r in hook.received]
    assert bodies[0]["event"] == "antares.check" and bodies[0]["verdict"] == "drifted"  # auto -> json
    assert bodies[1]["verdict"] == "insufficient_data"
    assert "blocks" in bodies[2] and bodies[2]["blocks"][0]["text"]["text"] == "Arm A"


def test_notify_dry_run_prints_instead_of_posting(hook, capsys):
    assert notify.notify(DRIFTED, hook.url, dry_run=True) is True
    assert hook.received == []
    out = json.loads(capsys.readouterr().out)
    assert out["event"] == "antares.check" and out["verdict"] == "drifted"


def test_notify_propagates_send_failures(hook):
    hook.status = 503
    with pytest.raises(notify.NotifyError, match="HTTP 503"):
        notify.notify(DRIFTED, hook.url)


def test_notify_error_only_for_failure_or_always(hook):
    assert notify.notify_error("boom", hook.url, lock="ghost", dataset="d") is False
    assert hook.received == []
    assert notify.notify_error("boom", hook.url, lock="ghost", dataset="d", when="failure") is True
    assert hook.received[0]["body"]["verdict"] == "error"
    assert hook.received[0]["body"]["error"] == "boom"


def test_example_result_is_a_passing_check():
    r = notify.example_result()
    assert r.verdict == "ok" and r.passed
    assert notify.render_text(r).startswith("OK example on your-org/so101_arm_a")
    json.dumps(notify.render_payload(r, "slack"))
