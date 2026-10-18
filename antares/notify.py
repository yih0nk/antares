"""Alert a webhook when :func:`antares.check` says a robot drifted.

The check is a CI gate; this module makes it monitoring. :func:`notify`
renders a :class:`~antares.check.CheckResult` as a compact alert and POSTs
it to a webhook URL: a Slack incoming webhook (``hooks.slack.com``) gets
Block Kit ``blocks``, anything else gets a generic JSON envelope
``{"event": "antares.check", ...}``. Standard library only (``urllib``),
so the CLI and the GitHub Action can post without extra dependencies.

A failed POST raises :class:`NotifyError`; callers decide whether that
matters (the CLI only warns, so a dead webhook never changes a verdict).
"""

from __future__ import annotations

import json
import math
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, TextIO

from antares.check import VERDICT_DRIFTED, VERDICT_INSUFFICIENT, VERDICT_OK, CheckResult

EVENT = "antares.check"
WHEN_CHOICES = ("drifted", "failure", "always")
FORMAT_CHOICES = ("auto", "slack", "json")
SLACK_HOST = "hooks.slack.com"
MAX_FAILED_JOINTS = 3  # named in the text; the rest are counted
MAX_SLACK_FIELDS = 10  # Slack's limit on a section's ``fields``
DEFAULT_TIMEOUT = 10.0


class NotifyError(RuntimeError):
    """The webhook could not be reached or answered with a non-2xx status."""


# -- rendering ------------------------------------------------------------------


def _deg(v: float | None) -> str:
    if v is None or not math.isfinite(v):
        return "n/a"
    return f"{v:.2f}"


def _verdict_word(result: CheckResult) -> str:
    if result.verdict == VERDICT_INSUFFICIENT:
        return "NO DATA"
    return "OK" if result.verdict == VERDICT_OK else "DRIFTED"


def _failed_joints(result: CheckResult) -> list[str]:
    """Joints over their limit, worst first."""
    failed = [j for j, jc in result.per_joint.items() if jc.limit is not None and not jc.passed]

    def excess(j: str) -> float:
        jc = result.per_joint[j]
        return jc.score / jc.limit if jc.limit else jc.score

    return sorted(failed, key=excess, reverse=True)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def headline(result: CheckResult) -> str:
    """``DRIFTED <lock> on <dataset>: 2.10 deg RMSE (limit 1.58)`` (or ``OK`` /
    ``NO DATA``)."""
    word = _verdict_word(result)
    where = f"{result.lock} on {result.dataset}"
    if result.verdict == VERDICT_INSUFFICIENT:
        return (
            f"{word} {where}: {result.n_episodes} episode(s), "
            f"need at least {result.min_episodes}"
        )
    return f"{word} {where}: {_deg(result.score)} deg RMSE (limit {_deg(result.limit)})"


def render_text(result: CheckResult, *, title: str | None = None) -> str:
    """A compact multi-line alert.

    First line is :func:`headline` (``title`` goes on its own line before
    it when given), then the worst joint, then up to three failed joints
    with ``score/limit``, then the episode count and the ISO time.
    """
    lines: list[str] = []
    if title:
        lines.append(title)
    lines.append(headline(result))
    if result.worst_joint is not None and result.worst_joint in result.per_joint:
        w = result.per_joint[result.worst_joint]
        lines.append(f"worst joint {result.worst_joint}: {_deg(w.score)} deg (limit {_deg(w.limit)})")
    failed = _failed_joints(result)
    if failed:
        parts = [f"{j} {_deg(result.per_joint[j].score)}/{_deg(result.per_joint[j].limit)}" for j in failed[:MAX_FAILED_JOINTS]]
        more = len(failed) - MAX_FAILED_JOINTS
        if more > 0:
            parts.append(f"+{more} more")
        lines.append(f"failed joints ({len(failed)}): " + ", ".join(parts))
    lines.append(f"{result.n_episodes} episodes, {result.n_frames} frames, {_timestamp()}")
    return "\n".join(lines)


def _slack_payload(result: CheckResult, text: str, title: str | None) -> dict[str, Any]:
    header = title or f"Antares check: {_verdict_word(result)}"
    body_lines = text.splitlines()
    if title:
        body_lines = body_lines[1:]  # the header block already carries it
    blocks: list[dict[str, Any]] = [
        {"type": "header", "text": {"type": "plain_text", "text": header[:150], "emoji": False}},
        {"type": "section", "text": {"type": "mrkdwn", "text": "*" + body_lines[0] + "*\n" + "\n".join(body_lines[1:])}},
    ]
    if result.per_joint:
        fields = []
        for j, jc in list(result.per_joint.items())[:MAX_SLACK_FIELDS]:
            status = ("PASS" if jc.passed else "FAIL") if jc.limit is not None else "n/a"
            fields.append({"type": "mrkdwn", "text": f"*{j}*\n{_deg(jc.score)} / {_deg(jc.limit)} {status}"})
        blocks.append({"type": "section", "fields": fields})
    return {"text": text, "blocks": blocks}


def render_payload(result: CheckResult, fmt: str, *, title: str | None = None) -> dict[str, Any]:
    """The JSON body for the webhook.

    ``fmt`` is ``"slack"`` (an incoming-webhook message with ``text`` and
    ``blocks``: a header, an mrkdwn section and a fields section for the
    joints) or ``"json"`` (``{"event": "antares.check", "verdict", "text",
    "result": result.to_dict()}``). Use :func:`resolve_format` to turn
    ``"auto"`` into one of them first.
    """
    text = render_text(result, title=title)
    if fmt == "slack":
        return _slack_payload(result, text, title)
    if fmt == "json":
        return {"event": EVENT, "verdict": result.verdict, "text": text, "result": result.to_dict()}
    raise ValueError(f"unknown notify format {fmt!r}; expected 'slack' or 'json'")


def render_error_payload(message: str, fmt: str, *, lock: str, dataset: str, title: str | None = None) -> dict[str, Any]:
    """Body for a check that could not run at all (no :class:`CheckResult`)."""
    text = f"ERROR {lock} on {dataset}: {message}\n{_timestamp()}"
    if title:
        text = f"{title}\n{text}"
    if fmt == "slack":
        header = title or "Antares check: ERROR"
        body = text.splitlines()[1:] if title else text.splitlines()
        return {
            "text": text,
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": header[:150], "emoji": False}},
                {"type": "section", "text": {"type": "mrkdwn", "text": "*" + body[0] + "*\n" + "\n".join(body[1:])}},
            ],
        }
    if fmt == "json":
        return {"event": EVENT, "verdict": "error", "text": text, "result": None,
                "error": message, "lock": lock, "dataset": dataset}
    raise ValueError(f"unknown notify format {fmt!r}; expected 'slack' or 'json'")


# -- policy ---------------------------------------------------------------------


def should_notify(result: CheckResult, when: str) -> bool:
    """``drifted``: only a drifted verdict; ``failure``: drifted or
    insufficient data; ``always``: every verdict."""
    if when not in WHEN_CHOICES:
        raise ValueError(f"unknown notify-when {when!r}; expected one of {', '.join(WHEN_CHOICES)}")
    if when == "always":
        return True
    if when == "failure":
        return result.verdict in (VERDICT_DRIFTED, VERDICT_INSUFFICIENT)
    return result.verdict == VERDICT_DRIFTED


def resolve_format(url: str, fmt: str = "auto") -> str:
    """``"auto"`` is ``"slack"`` for ``hooks.slack.com`` URLs, else ``"json"``."""
    if fmt not in FORMAT_CHOICES:
        raise ValueError(f"unknown notify format {fmt!r}; expected one of {', '.join(FORMAT_CHOICES)}")
    if fmt != "auto":
        return fmt
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    return "slack" if host == SLACK_HOST or host.endswith("." + SLACK_HOST) else "json"


# -- transport ------------------------------------------------------------------


def send(url: str, payload: dict[str, Any], timeout: float = DEFAULT_TIMEOUT) -> int:
    """POST ``payload`` as JSON and return the HTTP status.

    Raises :class:`NotifyError` when the URL is not http(s), the host cannot
    be reached, the request times out or the response is not 2xx.
    """
    scheme = urllib.parse.urlsplit(url).scheme.lower()
    if scheme not in ("http", "https"):
        raise NotifyError(f"webhook URL must be http(s), got {url!r}")
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "User-Agent": "antares-notify"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = int(resp.status)
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read(200).decode("utf-8", "replace").strip()
        except Exception:  # noqa: BLE001 - the status is the message
            pass
        raise NotifyError(f"webhook returned HTTP {e.code}" + (f": {detail}" if detail else "")) from e
    except urllib.error.URLError as e:
        raise NotifyError(f"could not reach webhook: {e.reason}") from e
    except (OSError, ValueError) as e:  # timeouts, refused connections, bad URLs
        raise NotifyError(f"could not reach webhook: {e}") from e
    if not 200 <= status < 300:
        raise NotifyError(f"webhook returned HTTP {status}")
    return status


def _emit(payload: dict[str, Any], out: TextIO | None) -> None:
    print(json.dumps(payload, indent=2), file=out or sys.stdout)


def notify(
    result: CheckResult,
    url: str,
    *,
    when: str = "drifted",
    fmt: str = "auto",
    title: str | None = None,
    dry_run: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
    out: TextIO | None = None,
) -> bool:
    """Post an alert for ``result`` to ``url`` when ``when`` says so.

    Returns True when a message was sent (or, with ``dry_run``, printed to
    ``out``/stdout instead of posted). Raises :class:`NotifyError` when the
    POST fails and ``ValueError`` for a bad ``when``/``fmt``.
    """
    if not should_notify(result, when):
        return False
    payload = render_payload(result, resolve_format(url, fmt), title=title)
    if dry_run:
        _emit(payload, out)
        return True
    send(url, payload, timeout=timeout)
    return True


def notify_error(
    message: str,
    url: str,
    *,
    lock: str,
    dataset: str,
    when: str = "drifted",
    fmt: str = "auto",
    title: str | None = None,
    dry_run: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
    out: TextIO | None = None,
) -> bool:
    """Alert that the check could not run (missing lock, unreadable dataset).

    Sent for ``when`` in ``failure``/``always``; ``drifted`` stays quiet
    because there is no verdict.
    """
    if when not in WHEN_CHOICES:
        raise ValueError(f"unknown notify-when {when!r}; expected one of {', '.join(WHEN_CHOICES)}")
    if when == "drifted":
        return False
    payload = render_error_payload(message, resolve_format(url, fmt), lock=lock, dataset=dataset, title=title)
    if dry_run:
        _emit(payload, out)
        return True
    send(url, payload, timeout=timeout)
    return True


def example_result(name: str = "example", dataset: str = "your-org/so101_arm_a") -> CheckResult:
    """A synthetic passing result for ``antares notify-test``."""
    from antares.check import JointCheck

    joints = {
        "shoulder_pan": 0.41, "shoulder_lift": 0.85, "elbow_flex": 1.20,
        "wrist_flex": 0.90, "wrist_roll": 0.70, "gripper": 0.30,
    }
    per_joint = {j: JointCheck(score=s, reference=1.0, limit=1.5, passed=True) for j, s in joints.items()}
    return CheckResult(
        passed=True,
        verdict=VERDICT_OK,
        score=1.07,
        reference=1.05,
        limit=1.58,
        per_joint=per_joint,
        worst_joint="elbow_flex",
        n_episodes=10,
        n_frames=3000,
        lock=name,
        dataset=dataset,
        episodes=list(range(10)),
        scored_joints=tuple(j for j in joints if j != "gripper"),
    )


__all__ = [
    "DEFAULT_TIMEOUT",
    "EVENT",
    "FORMAT_CHOICES",
    "WHEN_CHOICES",
    "NotifyError",
    "example_result",
    "headline",
    "notify",
    "notify_error",
    "render_error_payload",
    "render_payload",
    "render_text",
    "resolve_format",
    "send",
    "should_notify",
]
