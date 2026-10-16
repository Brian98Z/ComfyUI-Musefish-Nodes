"""ComfyUI API client: workflow submission, history polling, model/node discovery.

Talks to the headless ComfyUI instance that app.py supervises.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


class ComfyError(RuntimeError):
    pass


class ComfyClient:
    def __init__(self, host: str, port: int, timeout: float = 30.0):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout

    def _get(self, path: str, timeout: float | None = None) -> Any:
        req = urllib.request.Request(self.base + path)
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def alive(self) -> bool:
        try:
            self._get("/system_stats", timeout=3)
            return True
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return False

    def object_info(self, class_type: str | None = None) -> dict:
        path = "/object_info" if class_type is None else f"/object_info/{class_type}"
        return self._get(path, timeout=60)

    def upload(self, local_path: str, filename: str | None = None) -> str:
        """Upload an input file; returns the server-side subfolder-relative name."""
        import mimetypes
        import os
        import uuid

        name = filename or os.path.basename(local_path)
        boundary = uuid.uuid4().hex
        mime = mimetypes.guess_type(local_path)[0] or "application/octet-stream"
        with open(local_path, "rb") as f:
            payload = f.read()
        body = b"".join(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="image"; filename="{name}"\r\n'.encode(),
                f"Content-Type: {mime}\r\n\r\n".encode(),
                payload,
                f"\r\n--{boundary}--\r\n".encode(),
            ]
        )
        req = urllib.request.Request(
            self.base + "/upload/image",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.loads(r.read().decode("utf-8"))
        sub = resp.get("subfolder", "")
        return f"{sub}/{resp['name']}" if sub else resp["name"]

    def submit(self, prompt: dict, client_id: str) -> str:
        payload = {"prompt": prompt, "client_id": client_id}
        req = urllib.request.Request(
            self.base + "/prompt",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                resp = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            raise ComfyError(f"ComfyUI rejected the workflow: {detail}") from e
        return resp["prompt_id"]

    def history(self, prompt_id: str) -> dict:
        return self._get(f"/history/{prompt_id}", timeout=self.timeout)

    def queue(self) -> dict:
        return self._get("/queue", timeout=self.timeout)

    def wait(self, prompt_id: str, on_progress=None, poll: float = 1.0, timeout: float = 3600) -> dict:
        """Poll history until the prompt completes; returns its history entry."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            h = self.history(prompt_id).get(prompt_id)
            if h is not None:
                status = h.get("status", {})
                if status.get("status_str") == "error":
                    msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                    raise ComfyError(json.dumps(msgs, ensure_ascii=False)[:4000])
                return h
            if on_progress:
                try:
                    on_progress(self.queue())
                except Exception:
                    pass
            time.sleep(poll)
        raise ComfyError(f"timeout waiting for {prompt_id}")

    def download(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        from urllib.parse import urlencode

        qs = urlencode({"filename": filename, "subfolder": subfolder, "type": folder_type})
        req = urllib.request.Request(f"{self.base}/view?{qs}")
        with urllib.request.urlopen(req, timeout=1800) as r:
            return r.read()
