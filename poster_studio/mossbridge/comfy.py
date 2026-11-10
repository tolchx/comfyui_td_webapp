"""Cliente ComfyUI con urllib: salud, subida, encolado, espera y descarga."""

from __future__ import annotations

import json
import mimetypes
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


class EngineError(RuntimeError):
    pass


class ComfyClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8189, timeout: float = 20.0):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout

    def _request(self, path: str, data: bytes | None = None, headers: dict | None = None,
                 timeout: float | None = None):
        req = urllib.request.Request(self.base + path, data=data, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
            body = resp.read()
        return body

    # --- salud ---
    def stats(self) -> dict:
        return json.loads(self._request("/system_stats"))

    def alive(self) -> bool:
        try:
            self.stats()
            return True
        except Exception:
            return False

    def object_info(self) -> dict:
        return json.loads(self._request("/object_info", timeout=60))

    # --- entrada de archivos ---
    def upload(self, path: str, filename: str | None = None) -> str:
        filename = filename or path.split("/")[-1].split("\\")[-1]
        boundary = "----moss" + uuid.uuid4().hex
        with open(path, "rb") as fh:
            payload = fh.read()
        ctype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        parts = [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'.encode(),
            f"Content-Type: {ctype}\r\n\r\n".encode(),
            payload,
            f"\r\n--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue',
            f"\r\n--{boundary}--\r\n".encode(),
        ]
        body = b"".join(parts)
        raw = self._request("/upload/image", data=body,
                            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                            timeout=120)
        info = json.loads(raw)
        sub = info.get("subfolder") or ""
        name = info.get("name", filename)
        return f"{sub}/{name}" if sub else name

    # --- ejecución ---
    def submit(self, graph: dict) -> str:
        payload = json.dumps({"prompt": graph, "client_id": str(uuid.uuid4())}).encode()
        try:
            raw = self._request("/prompt", data=payload,
                                headers={"Content-Type": "application/json"}, timeout=60)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:800]
            raise EngineError(f"el motor rechazó el grafo ({exc.code}): {detail}") from exc
        return json.loads(raw)["prompt_id"]

    def wait(self, prompt_id: str, timeout: float = 900.0) -> dict:
        start = time.time()
        delay = 0.4
        while time.time() - start < timeout:
            time.sleep(delay)
            delay = min(delay * 1.15, 3.0)
            hist = json.loads(self._request(f"/history/{prompt_id}"))
            if prompt_id in hist:
                rec = hist[prompt_id]
                status = (rec.get("status") or {}).get("status_str")
                if status == "error":
                    raise EngineError("error de ejecución en el motor")
                files = [img["filename"] for node in rec.get("outputs", {}).values()
                         for img in node.get("images", [])]
                return {"seconds": round(time.time() - start, 1), "files": files}
        raise EngineError(f"timeout esperando el job {prompt_id}")

    def fetch(self, filename: str) -> bytes:
        query = urllib.parse.urlencode({"filename": filename, "type": "output"})
        return self._request(f"/view?{query}", timeout=180)

    def queue_depth(self) -> int:
        try:
            data = json.loads(self._request("/queue"))
            return len(data.get("queue_running", [])) + len(data.get("queue_pending", []))
        except Exception:
            return -1
