"""Thin HTTP client for the policy server, plus a RoboTwin-style adapter.

``MotusClient`` needs only numpy, msgpack, requests and ``protocol.py``. The
adapter functions ``get_model`` / ``eval`` / ``reset_model`` follow the
RoboTwin policy interface and show how an environment loop drives the
server; they contain no robot-specific control.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from protocol import (  # noqa: E402
    DEFAULT_TRANSPORT,
    Action,
    Observation,
    packb,
    protocol_descriptor,
)


class MotusClient:
    def __init__(self, url: str, *, transport: str = DEFAULT_TRANSPORT,
                 jpeg_quality: int = 95, timeout_s: float = 120.0):
        self.url = url.rstrip("/")
        self.transport = transport
        self.jpeg_quality = int(jpeg_quality)
        self.timeout_s = float(timeout_s)
        self.session = requests.Session()
        self.metadata: dict = {}
        self.session_epoch: Optional[str] = None
        self._seq = 0

    # -- transport ---------------------------------------------------------

    def _post(self, route: str, **kwargs) -> requests.Response:
        response = self.session.post(f"{self.url}/{route}", timeout=self.timeout_s, **kwargs)
        if not response.ok:
            try:
                detail = response.json().get("error")
            except ValueError:
                detail = response.text[:500]
            raise RuntimeError(f"/{route} failed ({response.status_code}): {detail}")
        return response

    def _observation(self, images: Dict[str, np.ndarray], state=None) -> bytes:
        self._seq += 1
        obs = Observation(
            images=images,
            state=None if state is None else np.asarray(state, dtype=np.float32),
            meta={"observation_id": f"{self.session_epoch}:{self._seq:08d}", "session_epoch": self.session_epoch},
        )
        return obs.to_bytes(self.transport, jpeg_quality=self.jpeg_quality)

    # -- API ---------------------------------------------------------------

    def handshake(self) -> dict:
        """Check the protocol and fetch the model contract (chunk length, cameras, ...)."""
        payload = self._post("handshake", json={"protocol": protocol_descriptor(self.transport)}).json()
        self.metadata = dict(payload["model"])
        return self.metadata

    def health(self) -> dict:
        response = self.session.get(f"{self.url}/healthz", timeout=self.timeout_s)
        response.raise_for_status()
        return response.json()

    def set_language(self, instruction: Optional[str] = None, embedding_file: Optional[str] = None) -> dict:
        """Encode ``instruction`` on the server, or upload a precomputed UMT5 embedding."""
        if embedding_file is not None:
            with open(embedding_file, "rb") as handle:
                data = {"instruction": instruction} if instruction else {}
                return self._post("set_language", data=data, files={"lang_embedding_file": handle}).json()
        return self._post("set_language", json={"instruction": instruction}).json()

    def reset(self) -> None:
        self.session_epoch = uuid.uuid4().hex
        self._seq = 0
        self._post("reset", json={"session_epoch": self.session_epoch})

    def update_obs(self, images: Dict[str, np.ndarray]) -> dict:
        """Send one cache-only frame taken while the current chunk executes."""
        return self._post("update_obs", data=self._observation(images)).json()

    def update_obs_batch(self, frames) -> dict:
        return self._post("update_obs_batch", data=packb([self._observation(images) for images in frames])).json()

    def infer(self, images: Dict[str, np.ndarray], state) -> np.ndarray:
        """Send the current frame and state; returns the action chunk [chunk_length, action_dim]."""
        response = self._post("infer", data=self._observation(images, state))
        return Action.from_bytes(response.content).actions

    def prepare_infer(self, images: Dict[str, np.ndarray]) -> str:
        """Split inference, step 1: send the frame early (VAE encode + cache update)."""
        return self._post("prepare_infer", data=self._observation(images)).json()["observation_id"]

    def predict(self, prepared_observation_id: str, state) -> np.ndarray:
        """Split inference, step 2: send the freshest state and get the action chunk."""
        payload = {"prepared_observation_id": prepared_observation_id,
                   "state": np.asarray(state, dtype=np.float32).tolist()}
        return Action.from_bytes(self._post("predict", json=payload).content).actions

    def finish(self) -> None:
        self._post("finish")

    def close(self) -> None:
        self.session.close()


# ---------------------------------------------------------------------------
# RoboTwin-style adapter
# ---------------------------------------------------------------------------
#
# Expected observation layout (RoboTwin convention):
#   observation["observation"][camera]["rgb"]  HxWx3 uint8 RGB, for every camera
#                                              listed in the server's vision.cameras
#   observation["joint_action"]["vector"]      robot state in the model's state space
# The environment exposes get_instruction(), get_obs() and take_action(action, ...).

def _images(observation, cameras) -> Dict[str, np.ndarray]:
    return {name: np.asarray(observation["observation"][name]["rgb"], dtype=np.uint8) for name in cameras}


def _state(observation) -> np.ndarray:
    return np.asarray(observation["joint_action"]["vector"], dtype=np.float32)


class _Policy:
    def __init__(self, client: MotusClient, action_type: str):
        self.client = client
        self.action_type = action_type
        self.instruction: Optional[str] = None
        self.started = False


def get_model(usr_args):
    """usr_args: server_url (required), transport, jpeg_quality, action_type."""
    client = MotusClient(
        usr_args["server_url"],
        transport=usr_args.get("transport", DEFAULT_TRANSPORT),
        jpeg_quality=int(usr_args.get("jpeg_quality", 95)),
    )
    client.handshake()
    return _Policy(client, usr_args.get("action_type", "qpos"))


def eval(TASK_ENV, model, observation):
    """Predict one chunk for ``observation`` and execute it in the environment.

    The first call of an episode primes the server's cache with this frame.
    While the chunk executes, a frame is taken every ``actions_per_frame``
    actions; all but the last go to /update_obs, and the last one is the
    ``observation`` the environment passes to the next ``eval`` call.
    """
    client, meta = model.client, model.client.metadata
    if not model.started:
        instruction = TASK_ENV.get_instruction()
        if instruction != model.instruction:
            client.set_language(instruction)
            model.instruction = instruction
        client.reset()
        model.started = True

    actions = client.infer(_images(observation, meta["cameras"]), _state(observation))

    interval = int(meta["actions_per_frame"])
    updates = int(meta["cache_updates_per_chunk"])
    for index, action in enumerate(actions, start=1):
        # Replace with the environment's own command interface; actions are in
        # the model's action space (see README "Tensor conventions").
        TASK_ENV.take_action(action, action_type=model.action_type)
        if index % interval == 0 and index // interval <= updates:
            client.update_obs(_images(TASK_ENV.get_obs(), meta["cameras"]))


def reset_model(model):
    """Called by the environment loop between episodes."""
    model.started = False
