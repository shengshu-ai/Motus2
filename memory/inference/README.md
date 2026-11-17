# Inference server

A machine-independent policy server for Motus2 Memory checkpoints and a
client example. The server runs on a GPU host and owns the model and its streaming
KV cache; a client on the robot or simulator side sends camera frames and
state and executes the returned action chunks.

```text
client (robot / simulator)                 server (GPU host)
  cameras + state  --HTTP, msgpack-->   runtime.MotusRuntime
  execute actions  <--action chunk---     models/ (release model code)
```

| File | Role |
|------|------|
| `server.py` | Flask app and CLI |
| `runtime.py` | model build/load, frame composition, memory KV cache, prediction |
| `protocol.py` | `Observation` / `Action` wire format, image encode/decode |
| `client.py` | `MotusClient` and RoboTwin-style `get_model` / `eval` / `reset_model` |
| `encode_instruction.py` | precompute a UMT5 instruction embedding |
| `config_example.yaml` | server config |

The runtime imports the training-side model (`models/`, `train/train.py`);
there is no separate copy of the model code.

## Run

```bash
pip install -r requirements.txt
python inference/server.py --config inference/config_example.yaml   # GPU host
curl -s http://127.0.0.1:8000/healthz
```

`model.training_config` must be the YAML the checkpoint was trained with; the
model is built from it with `create_model_and_optimizer` (optimizer unused)
and loaded with `load_checkpoint_weights`.

## Request sequence

```text
POST /handshake      {"protocol": protocol_descriptor(transport)}   -> model contract
POST /set_language   {"instruction": "..."} or an uploaded embedding  (resets the episode)
POST /reset          {"session_epoch": "<new id>"}
POST /infer          frame 0 + state                                   -> chunk 0
loop:
    execute the chunk; every `actions_per_frame` actions take a frame
    POST /update_obs for the first `cache_updates_per_chunk` (= 4K-1) frames
    POST /infer      with the last frame (after the final action) + state -> next chunk
POST /finish
```

`/infer` may be split into `/prepare_infer` (frame; VAE encode and cache
update) and `/predict` (`{"prepared_observation_id", "state"}`), so the state
can be sampled after the frame has been processed. `/update_obs_batch`
accepts a msgpack list of observations.

The server enforces the schedule: `/update_obs` before the first `/infer` is
rejected, and every later `/infer` needs exactly `4K - 1` preceding
`/update_obs` frames. `observation_id` must be unique within an episode and
`session_epoch` must match the last `/reset`.

## Messages

Observation (msgpack): `{"magic", "version", "transport", "images":
{camera: packet}, "state": ndarray | null, "meta": {"observation_id",
"session_epoch"}}`. A packet is `{"encoding": "raw_rgb8", "shape", "dtype",
"data"}` or `{"encoding": "jpeg" | "png", "data"}`. Action (msgpack):
`{"actions": ndarray [chunk_length, action_dim] float32, "meta"}`. The
handshake returns `action_dim`, `state_dim`, `chunk_length`,
`frames_per_chunk`, `cache_updates_per_chunk`, `actions_per_frame`,
`cameras`, `layout`, `num_views`, `video_size` and the `memory` settings.

## Tensor conventions

- Images: HxWx3 uint8 RGB per camera, named as in `vision.cameras`. The
  server composes them with `vision.layout` (`views`, `grid` or `tshape`),
  letterboxes to `(common.video_height, common.video_width)` and scales to
  [0, 1]. The layout and camera order must match how the training latents
  were produced.
- The model is single-view: all cameras are composed into one frame.
- Frames are taken at video rate: one frame every `actions_per_frame =
  video_action_freq_ratio * action_upsample_ratio` actions. The first frame
  of an episode is encoded alone into one latent; every later group of 4
  frames becomes one latent.
- Every latent is appended to the KV cache for the whole episode; the model
  keeps the first `memory.anchor_latents` latents, all gist tokens and the
  newest `memory.recent_latents` latents visible. The runtime never trims the
  cache, since anchors and the absolute RoPE clock span the episode.
- State: `[state_dim]` float in the model's state space, i.e. exactly what
  the training loader feeds as `states` (for `joint`, the 14-D joint vector;
  for `qpos20_rel6d`, dataset-normalized hand qpos plus absolute wrist
  xyz in meters and rot6d).
- Actions: `[K * actions_per_latent, action_dim]` from
  `Motus.inference_step_streaming`, passed through
  `Motus._unscale_qpos20_rel_wrist_xyz` (the streaming call returns
  normalized wrist xyz). `joint` actions are absolute joint targets. For
  `qpos20_rel6d` the wrist part is relative to the state anchor (xyz in
  meters, rot6d) and hand qpos is in the dataset-normalized space;
  `utils/relative_eef` converts back to absolute poses.

## Out of scope

Robot control, camera acquisition, time alignment between sensors, safety
limits and conversion of model actions to a specific robot's commands are
deployment-specific and not part of this package. The server has no
authentication; run it on a trusted network.
