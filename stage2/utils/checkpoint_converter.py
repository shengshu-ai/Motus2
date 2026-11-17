"""Checkpoint key conversion utilities for different model formats."""

import logging

logger = logging.getLogger(__name__)


def convert_diffusers_keys(sd: dict) -> dict:
    """Convert diffusers format keys to WanModel format.

    Complete mapping (verified by shape matching):
    - condition_embedder.text_embedder.linear_1 -> text_embedding.0
    - condition_embedder.text_embedder.linear_2 -> text_embedding.2
    - condition_embedder.time_embedder.linear_1 -> time_embedding.0
    - condition_embedder.time_embedder.linear_2 -> time_embedding.2
    - condition_embedder.time_proj -> time_projection.1
    - blocks.*.attn1.* -> blocks.*.self_attn.*
    - blocks.*.attn2.* -> blocks.*.cross_attn.*
    - blocks.*.ffn.net.0.proj -> blocks.*.ffn.0
    - blocks.*.ffn.net.2 -> blocks.*.ffn.2
    - blocks.*.norm2 -> blocks.*.norm3
    - blocks.*.norm_out -> blocks.*.norm3
    - blocks.*.scale_shift_table -> blocks.*.modulation
    - proj_out -> head.head
    - scale_shift_table -> head.modulation

    Args:
        sd: State dict from diffusers format checkpoint

    Returns:
        Converted state dict compatible with WanModel format
    """
    converted = {}
    for k, v in sd.items():
        new_key = k

        # Skip condition_embedder_action (not used in WanModel)
        if k.startswith("condition_embedder_action"):
            continue

        # Convert condition_embedder.text_embedder
        if k.startswith("condition_embedder.text_embedder.linear_1"):
            new_key = k.replace("condition_embedder.text_embedder.linear_1", "text_embedding.0")
        elif k.startswith("condition_embedder.text_embedder.linear_2"):
            new_key = k.replace("condition_embedder.text_embedder.linear_2", "text_embedding.2")
        # Convert condition_embedder.time_embedder
        elif k.startswith("condition_embedder.time_embedder.linear_1"):
            new_key = k.replace("condition_embedder.time_embedder.linear_1", "time_embedding.0")
        elif k.startswith("condition_embedder.time_embedder.linear_2"):
            new_key = k.replace("condition_embedder.time_embedder.linear_2", "time_embedding.2")
        # Convert condition_embedder.time_proj -> time_projection.1
        elif k.startswith("condition_embedder.time_proj"):
            new_key = k.replace("condition_embedder.time_proj", "time_projection.1")
        # Skip condition_embedder.act_fn (SiLU is hardcoded in WanModel)
        elif k.startswith("condition_embedder.act_fn"):
            continue
        # Skip condition_embedder.timesteps_proj (not used in WanModel)
        elif k.startswith("condition_embedder.timesteps_proj"):
            continue
        # Convert blocks.*.attn1.* -> blocks.*.self_attn.*
        elif ".attn1." in k:
            new_key = k.replace(".attn1.to_q.", ".self_attn.q.")
            new_key = new_key.replace(".attn1.to_k.", ".self_attn.k.")
            new_key = new_key.replace(".attn1.to_v.", ".self_attn.v.")
            new_key = new_key.replace(".attn1.to_out.0.", ".self_attn.o.")
            new_key = new_key.replace(".attn1.norm_q.", ".self_attn.norm_q.")
            new_key = new_key.replace(".attn1.norm_k.", ".self_attn.norm_k.")
        # Convert blocks.*.attn2.* -> blocks.*.cross_attn.*
        elif ".attn2." in k:
            new_key = k.replace(".attn2.to_q.", ".cross_attn.q.")
            new_key = new_key.replace(".attn2.to_k.", ".cross_attn.k.")
            new_key = new_key.replace(".attn2.to_v.", ".cross_attn.v.")
            new_key = new_key.replace(".attn2.to_out.0.", ".cross_attn.o.")
            new_key = new_key.replace(".attn2.norm_q.", ".cross_attn.norm_q.")
            new_key = new_key.replace(".attn2.norm_k.", ".cross_attn.norm_k.")
        # Convert blocks.*.ffn.net.0.proj -> blocks.*.ffn.0
        elif ".ffn.net.0.proj." in k:
            new_key = k.replace(".ffn.net.0.proj.", ".ffn.0.")
        # Convert blocks.*.ffn.net.2 -> blocks.*.ffn.2
        elif ".ffn.net.2." in k:
            new_key = k.replace(".ffn.net.2.", ".ffn.2.")
        # Convert blocks.*.norm2 -> blocks.*.norm3
        elif ".norm2." in k:
            new_key = k.replace(".norm2.", ".norm3.")
        # Convert blocks.*.norm_out -> blocks.*.norm3
        elif ".norm_out." in k:
            new_key = k.replace(".norm_out.", ".norm3.")
        # Convert blocks.*.scale_shift_table -> blocks.*.modulation
        elif ".scale_shift_table" in k:
            new_key = k.replace(".scale_shift_table", ".modulation")
        # Convert proj_out -> head.head
        elif k.startswith("proj_out"):
            new_key = k.replace("proj_out", "head.head")
        # Convert scale_shift_table -> head.modulation (top-level)
        elif k.startswith("scale_shift_table"):
            new_key = k.replace("scale_shift_table", "head.modulation")
        # Skip action_embedder (not used in WanModel for video-only training)
        elif k.startswith("action_embedder"):
            continue
        # Skip action_proj_out (not used in WanModel)
        elif k.startswith("action_proj_out"):
            continue
        # Skip patch_embedding_mlp (diffusers format should use patch_embedding, not patch_embedding_mlp)
        elif k.startswith("patch_embedding_mlp"):
            continue

        converted[new_key] = v
    return converted


def strip_dit_prefix(sd: dict) -> dict:
    """Strip 'dit.' prefix from checkpoint keys if present.

    Args:
        sd: State dict that may contain 'dit.' prefixed keys

    Returns:
        State dict with 'dit.' prefix removed
    """
    if not any(k.startswith("dit.") for k in sd):
        return sd
    return {(k[4:] if k.startswith("dit.") else k): v for k, v in sd.items()}
