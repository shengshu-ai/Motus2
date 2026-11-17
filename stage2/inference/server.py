#!/usr/bin/env python3
"""HTTP policy server: one model, one episode at a time.

Routes (JSON unless noted; errors are ``{"error": ...}`` with status 4xx/5xx):

  POST /handshake        {"protocol": descriptor} -> {"ok", "protocol", "model"}
  GET  /healthz          -> {"ok", ...runtime health}
  POST /set_language     {"instruction"} | {"lang_embedding_path"} | multipart
                         file "lang_embedding_file"; resets the episode
  POST /reset            {"session_epoch"} -> {"status": "ok"}
  POST /update_obs       body: Observation bytes (cache-only frame)
  POST /update_obs_batch body: msgpack list of Observation bytes
  POST /infer            body: Observation bytes with state -> Action bytes
  POST /prepare_infer    body: Observation bytes (state not needed)
                         -> {"status": "prepared", "observation_id"}
  POST /predict          {"prepared_observation_id", "state"} -> Action bytes
  POST /finish           ends the episode -> {"status": "ok"}

Usage:
  python inference/server.py --config inference/config_example.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parent))

from protocol import (  # noqa: E402
    Observation,
    ProtocolError,
    normalize_transport,
    protocol_descriptor,
    unpackb,
    validate_protocol_descriptor,
)

logger = logging.getLogger(__name__)


def create_app(runtime, transport: str = "raw_rgb8") -> Flask:
    """Wrap a runtime exposing metadata/health/reset/set_language/accept/
    update_obs/prepare/set_state/predict/infer (see runtime.MotusRuntime)."""
    transport = normalize_transport(transport)
    app = Flask(__name__)

    def _error(exc: Exception, status: int = 400):
        logger.warning("%s rejected: %s", request.path, exc)
        return jsonify({"error": str(exc)}), status

    def _decode() -> Observation:
        return Observation.from_bytes(request.get_data(), expected_transport=transport)

    def _action_response(action, started: float) -> Response:
        resp = Response(action.to_bytes(), mimetype="application/octet-stream")
        resp.headers["X-Inference-Time"] = f"{time.time() - started:.6f}"
        return resp

    @app.route("/handshake", methods=["POST"])
    def handshake():
        payload = request.get_json(silent=True) or {}
        try:
            validate_protocol_descriptor(payload.get("protocol") or {}, transport)
        except ProtocolError as exc:
            return jsonify({"error": str(exc), "protocol": protocol_descriptor(transport)}), 409
        return jsonify({"ok": True, "protocol": protocol_descriptor(transport), "model": runtime.metadata()})

    @app.route("/healthz", methods=["GET"])
    def healthz():
        return jsonify({"ok": True, **runtime.health()})

    @app.route("/set_language", methods=["POST"])
    def set_language():
        payload = dict(request.form.to_dict()) if request.form else {}
        payload.update(request.get_json(silent=True) or {})
        uploaded = request.files.get("lang_embedding_file")
        try:
            result = runtime.set_language(
                instruction=payload.get("instruction"),
                embedding_bytes=None if uploaded is None else uploaded.read(),
                embedding_path=payload.get("lang_embedding_path"),
            )
        except Exception as exc:  # noqa: BLE001
            return _error(exc)
        return jsonify({"status": "ok", "cache_reset": True, **result})

    @app.route("/reset", methods=["POST"])
    def reset():
        payload = request.get_json(silent=True) or {}
        runtime.reset(session_epoch=payload.get("session_epoch"))
        return jsonify({"status": "ok"})

    @app.route("/finish", methods=["POST"])
    def finish():
        runtime.reset()
        return jsonify({"status": "ok"})

    @app.route("/update_obs", methods=["POST"])
    def update_obs():
        try:
            obs = _decode()
            runtime.accept(obs, cache_only=True)
            runtime.update_obs(obs)
        except (ProtocolError, ValueError, RuntimeError) as exc:
            return _error(exc)
        return jsonify({"status": "ok", **runtime.health()})

    @app.route("/update_obs_batch", methods=["POST"])
    def update_obs_batch():
        try:
            items = unpackb(request.get_data())
            if not isinstance(items, list):
                raise ProtocolError("update_obs_batch body must be a msgpack list")
            for data in items:
                obs = Observation.from_bytes(data, expected_transport=transport)
                runtime.accept(obs, cache_only=True)
                runtime.update_obs(obs)
        except (ProtocolError, ValueError, RuntimeError) as exc:
            return _error(exc)
        return jsonify({"status": "ok", "count": len(items), **runtime.health()})

    @app.route("/infer", methods=["POST"])
    def infer():
        started = time.time()
        try:
            obs = _decode()
            runtime.accept(obs, cache_only=False)
            action = runtime.infer(obs)
        except (ProtocolError, ValueError, RuntimeError) as exc:
            return _error(exc)
        return _action_response(action, started)

    @app.route("/prepare_infer", methods=["POST"])
    def prepare_infer():
        try:
            obs = _decode()
            observation_id = runtime.accept(obs, cache_only=False)
            runtime.prepare(obs)
        except (ProtocolError, ValueError, RuntimeError) as exc:
            return _error(exc)
        return jsonify({"status": "prepared", "observation_id": observation_id})

    @app.route("/predict", methods=["POST"])
    def predict():
        started = time.time()
        payload = request.get_json(silent=True) or {}
        try:
            runtime.set_state(payload.get("state"), str(payload.get("prepared_observation_id") or ""))
            action = runtime.predict()
        except (ValueError, RuntimeError) as exc:
            return _error(exc)
        return _action_response(action, started)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Motus2 policy server")
    parser.add_argument("--config", required=True, help="inference config (see config_example.yaml)")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from runtime import MotusRuntime, load_server_config

    cfg = load_server_config(args.config)
    server_cfg = dict(cfg.get("server") or {})
    host = args.host or str(server_cfg.get("host", "127.0.0.1"))
    port = args.port or int(server_cfg.get("port", 8000))
    runtime = MotusRuntime(cfg)
    app = create_app(runtime, transport=str((cfg.get("protocol") or {}).get("transport", "raw_rgb8")))
    logger.info("serving on %s:%d", host, port)
    # Single-threaded: requests touch one model and one episode in order.
    app.run(host=host, port=port, threaded=False)


if __name__ == "__main__":
    main()
