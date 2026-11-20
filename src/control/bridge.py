"""Versioned line protocol and process lifecycle for the live ns-3 simulator."""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import re
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config import RecoveryConfig


class ProtocolError(RuntimeError):
    """A simulator message or action violated the versioned protocol."""


class SimulatorTimeoutError(RuntimeError):
    """The simulator did not produce its next frame in time."""


SimulatorTimeout = SimulatorTimeoutError


def _is_wsl_launcher(launcher: Sequence[str]) -> bool:
    return bool(launcher) and Path(launcher[0]).name.lower() in {"wsl", "wsl.exe"}


def _wsl_path(path: Path, distribution: str | None = None) -> str:
    command = ["wsl.exe"]
    if distribution:
        command += ["-d", distribution]
    command += ["--exec", "wslpath", "-a", str(path.resolve()).replace("\\", "/")]
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True, timeout=15).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Could not translate path for WSL: {path}") from exc


def _validate_manifest(binary: Path) -> None:
    """Check declared source and binary provenance for repository recovery builds."""
    for root in binary.parents:
        if (root / "pyproject.toml").is_file() and (root / "simulation" / "ns3").is_dir():
            break
    else:
        return
    recovery_dir = root / "build" / "recovery"
    if not binary.is_relative_to(recovery_dir):
        return
    manifest = recovery_dir / "source.sha256"
    if not manifest.is_file():
        raise RuntimeError(f"Missing recovery build manifest: {manifest}")
    entries = manifest.read_text(encoding="utf-8").splitlines()
    if not entries:
        raise RuntimeError(f"Empty recovery build manifest: {manifest}")
    seen_binary = False
    for entry in entries:
        match = re.fullmatch(r"([0-9a-fA-F]{64})  (.+)", entry)
        if match is None:
            raise RuntimeError(f"Malformed recovery build manifest line: {entry}")
        expected, relative = match.groups()
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise RuntimeError(f"Missing or unsafe recovery build input: {relative}")
        if hashlib.sha256(candidate.read_bytes()).hexdigest().lower() != expected.lower():
            raise RuntimeError(f"Stale recovery build input: {relative}")
        if candidate == binary:
            seen_binary = True
    if not seen_binary:
        raise RuntimeError(f"Recovery binary omitted from manifest: {binary}")


def make_launcher(binary: Path, wsl_distribution: str | None = None) -> list[str]:
    """Build an argv prefix; WSL paths are translated without shell quoting."""
    binary = Path(binary).resolve()
    _validate_manifest(binary)
    if wsl_distribution is None:
        if os.name == "nt" and binary.suffix.lower() != ".exe":
            raise ValueError("A WSL distribution is required for a Linux simulator on Windows")
        return [str(binary)]
    return ["wsl.exe", "-d", wsl_distribution, "--exec", _wsl_path(binary, wsl_distribution)]


def simulator_output_path(path: Path, launcher: Sequence[str]) -> str:
    if not _is_wsl_launcher(launcher):
        return str(path.resolve())
    distribution = None
    if "-d" in launcher:
        distribution = launcher[launcher.index("-d") + 1]
    return _wsl_path(path, distribution)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ProtocolError(f"{name} must be a finite number")
    return float(value)


def _matrix(frame: Mapping[str, Any], name: str, n: int, *, boolean: bool = False) -> None:
    matrix = frame[name]
    if not isinstance(matrix, list) or len(matrix) != n:
        raise ProtocolError(f"{name} must be {n}x{n}")
    for row in matrix:
        if not isinstance(row, list) or len(row) != n:
            raise ProtocolError(f"{name} must be {n}x{n}")
        for value in row:
            if boolean:
                if not isinstance(value, bool) and not (type(value) is int and value in (0, 1)):
                    raise ProtocolError(f"{name} must contain booleans or 0/1")
            else:
                _finite_number(value, name)


def validate_frame(frame: Any, config: RecoveryConfig, expected_step: int, *, require_service_observation=False, previous=None) -> dict[str, Any]:
    if not isinstance(frame, dict):
        raise ProtocolError("Frame must be a JSON object")
    fields = {
        "version",
        "step",
        "positions",
        "velocities",
        "travel",
        "rx_ratio",
        "rssi",
        "rssi_mask",
        "components",
        "source_dest_connected",
        "olsr_route_usable",
        "connected_streak",
        "safety_rejected",
        "terminal",
    }
    missing = fields - frame.keys()
    if missing:
        raise ProtocolError(f"Missing frame fields: {sorted(missing)}")
    if type(frame["version"]) is not int or frame["version"] != 1:
        raise ProtocolError("Unsupported protocol version")
    if type(frame["step"]) is not int or frame["step"] != expected_step:
        raise ProtocolError(f"Expected frame step {expected_step}")
    n = config.num_uavs
    for name in ("positions", "velocities"):
        rows = frame[name]
        if not isinstance(rows, list) or len(rows) != n:
            raise ProtocolError(f"{name} must have {n} vectors")
        for row in rows:
            if not isinstance(row, list) or len(row) != 3:
                raise ProtocolError(f"{name} must contain 3D vectors")
            for value in row:
                _finite_number(value, name)
    travel = frame["travel"]
    if not isinstance(travel, list) or len(travel) != n:
        raise ProtocolError(f"travel must have {n} entries")
    for value in travel:
        if _finite_number(value, "travel") < 0:
            raise ProtocolError("travel must be nonnegative")
    for name in ("rx_ratio", "rssi"):
        _matrix(frame, name, n)
    _matrix(frame, "rssi_mask", n, boolean=True)
    frame["rssi_mask"] = [[bool(value) for value in row] for row in frame["rssi_mask"]]
    for row in frame["rx_ratio"]:
        for value in row:
            if not 0 <= value <= 1:
                raise ProtocolError("rx_ratio outside [0,1]")
    if type(frame["components"]) is not int or not 1 <= frame["components"] <= n:
        raise ProtocolError("Invalid component count")
    if type(frame["connected_streak"]) is not int or frame["connected_streak"] < 0:
        raise ProtocolError("Invalid connected streak")
    for name in ("source_dest_connected", "olsr_route_usable", "safety_rejected", "terminal"):
        value = frame[name]
        if not isinstance(value, bool) and not (type(value) is int and value in (0, 1)):
            raise ProtocolError(f"{name} must be boolean or 0/1")
        frame[name] = bool(value)
    if require_service_observation:
        from .path_service.observation import validate_service_observation
        try: validate_service_observation(frame, config, previous)
        except ValueError as exc: raise ProtocolError(str(exc)) from exc
    return frame


class SimulatorBridge:
    def __init__(self, command: Sequence[str], config: RecoveryConfig, stderr_path: Path, timeout_s: float = 30.0, stdout_path: Path | None = None):
        self.config = config
        self.timeout_s = timeout_s
        self.stderr_path = Path(stderr_path)
        self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr = self.stderr_path.open("w", encoding="utf-8")
        self._stdout_log = Path(stdout_path).open('x', encoding='utf-8') if stdout_path else None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._reader_error: Exception | None = None
        self.process: subprocess.Popen[str] | None = None
        self._closed = False
        try:
            self.process = subprocess.Popen(
                list(command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
            self._reader = threading.Thread(target=self._read_stdout, daemon=True)
            self._reader.start()
        except BaseException:
            self._stderr.close()
            if self._stdout_log is not None: self._stdout_log.close()
            raise

    def _read_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            for line in self.process.stdout:
                if self._stdout_log is not None:
                    self._stdout_log.write(line); self._stdout_log.flush()
                self._lines.put(line.rstrip("\r\n"))
        except Exception as exc:
            self._reader_error = exc
        finally:
            # The reader owns both stdout handles. Child exit can precede
            # buffered-pipe drain; another thread must never close them while
            # this iterator is still reading or the log is still being written.
            for stream in (self.process.stdout,self._stdout_log):
                if stream is not None:
                    try: stream.close()
                    except Exception as exc:
                        if self._reader_error is None: self._reader_error = exc
            self._lines.put(None)

    def next_frame(self, expected_step: int) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                line = self._lines.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty as exc:
                self.close(force=True)
                raise SimulatorTimeout(
                    f"No UAVRL frame for step {expected_step} within {self.timeout_s}s; stderr: {self.stderr_path}"
                ) from exc
            if line is None:
                self.close(force=True)
                raise ProtocolError(f"Simulator exited before frame {expected_step}; stderr: {self.stderr_path}")
            if line.startswith("UAVRL "):
                break
            # ns-3 can print ordinary diagnostics on stdout. They never count as
            # protocol frames and cannot extend the frame deadline.
            if time.monotonic() >= deadline:
                self.close(force=True)
                raise SimulatorTimeout(
                    f"No UAVRL frame for step {expected_step} within {self.timeout_s}s; stderr: {self.stderr_path}"
                )
        try:
            frame = json.loads(line[6:], parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            return validate_frame(frame, self.config, expected_step)
        except (ValueError, ProtocolError) as exc:
            self.close(force=True)
            raise ProtocolError(f"Invalid frame {expected_step}: {exc}") from exc

    def send_action(self, step: int, action: Sequence[float]) -> None:
        expected = 3 * len(self.config.relay_ids)
        if len(action) != expected:
            raise ValueError(f"Action must contain {expected} normalized velocity values")
        values = [_finite_number(float(value), "action") for value in action]
        if any(abs(value) > 1 for value in values):
            raise ValueError("Action values must lie in [-1,1]")
        if self.process is None or self.process.poll() is not None or self.process.stdin is None:
            self.close(force=True)
            raise ProtocolError("Simulator is not running")
        try:
            self.process.stdin.write(f"UAVACT 1 {step} " + " ".join(format(v, ".9g") for v in values) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self.close(force=True)
            raise ProtocolError("Could not send action to simulator") from exc

    def close(self, *, force: bool = False) -> None:
        if self._closed:
            return
        process = self.process
        if process is None:
            return
        if process.poll() is None and not force and process.stdin is not None:
            try:
                process.stdin.write("UAVQUIT\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
        try:
            process.wait(timeout=self.timeout_s if not force else 1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        finally:
            if process.stdin is not None:
                process.stdin.close()
            self._stderr.close()
            self._reader.join(timeout=5)
            self._closed = True
        if self._reader.is_alive():
            raise ProtocolError(f"Simulator stdout reader did not finish draining; stderr: {self.stderr_path}")
        if self._reader_error is not None:
            raise ProtocolError(f"Simulator stdout read/log failure: {type(self._reader_error).__name__}: {self._reader_error}; stderr: {self.stderr_path}") from self._reader_error
        if not force and process.returncode != 0:
            raise ProtocolError(f"Simulator exited with code {process.returncode}; stderr: {self.stderr_path}")
