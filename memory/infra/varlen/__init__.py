"""Open-source-safe varlen packing and attention helpers."""

from .types import PackedStreams, VarlenBatchMeta, VarlenLayout, VarlenRopeMeta, VarlenSegment, VarlenStreamMeta
from .packing import (
    pack_streams,
    unpack_streams,
    pad_to_length,
    pack_named_streams,
    unpack_named_streams,
    DEFAULT_STREAM_ORDER,
)
from .attention import varlen_block_attention, register_attention_backend, available_attention_backends
from .block_mask import build_packed_memory_window_block_mask, clear_block_mask_cache, block_mask_cache_info
from .preprocessor import build_memory_varlen_batch
from .rope import assemble_packed_freqs
from .stream_major import (
    StreamMajorPlan,
    build_stream_major_plan,
    to_stream_major,
    to_sample_major,
    remap_dense_mask,
)
from .factored_plans import SegmentFactoredPlans, build_segment_factored_plans
from .factored_attention import (
    HAS_FLASH_ATTN,
    SampleMajorLayoutMeta,
    build_sample_major_layout_meta,
    factored_memory_attention,
)
from .pad_ceilings import get_pad_ceilings, grow_pad_ceilings, reset_pad_ceilings

__all__ = [
    "PackedStreams",
    "VarlenBatchMeta",
    "VarlenLayout",
    "VarlenRopeMeta",
    "VarlenSegment",
    "VarlenStreamMeta",
    "pack_streams",
    "unpack_streams",
    "pad_to_length",
    "pack_named_streams",
    "unpack_named_streams",
    "DEFAULT_STREAM_ORDER",
    "varlen_block_attention",
    "register_attention_backend",
    "available_attention_backends",
    "build_packed_memory_window_block_mask",
    "build_memory_varlen_batch",
    "clear_block_mask_cache",
    "block_mask_cache_info",
    "assemble_packed_freqs",
    "StreamMajorPlan",
    "build_stream_major_plan",
    "to_stream_major",
    "to_sample_major",
    "remap_dense_mask",
    "SegmentFactoredPlans",
    "build_segment_factored_plans",
    "SampleMajorLayoutMeta",
    "build_sample_major_layout_meta",
    "factored_memory_attention",
    "HAS_FLASH_ATTN",
    "grow_pad_ceilings",
    "reset_pad_ceilings",
    "get_pad_ceilings",
]
