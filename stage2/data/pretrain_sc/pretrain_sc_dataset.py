"""Scaling-law pretrain dataset.

This wraps :class:`data.pretrain.pretrain_dataset.PretrainDataset` with two
extra pieces of sampling metadata:

- deterministic nested subsets by source using a stable hash rank;
- per-source sampling weights, usually proportional to target hours.
"""

import hashlib
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from data.pretrain.pretrain_dataset import PretrainDataset, _normalize_remote_path, _strip_remote_scheme

logger = logging.getLogger(__name__)


class PretrainSCDataset(PretrainDataset):
    """Pretrain dataset for scaling-law runs with source/hour-aware sampling."""

    def __init__(
        self,
        subset_fraction: float = 1.0,
        subset_seed: int = 0,
        temporal_buckets: Optional[List[Dict[str, Any]]] = None,
        batch_size_by_window: Optional[Dict[str, int]] = None,
        sampling_weight: str = "hours",
        latent_seconds: float = 0.8,
        num_batches_per_epoch: Optional[int] = None,
        dynamic_even_window: bool = True,
        min_latent_window: int = 4,
        max_total_latent_frames: Optional[int] = None,
        **kwargs,
    ):
        self.subset_fraction = float(subset_fraction)
        if self.subset_fraction <= 0 or self.subset_fraction > 1:
            raise ValueError(f"subset_fraction must be in (0, 1], got {subset_fraction}")
        self.subset_seed = int(subset_seed)
        self.latent_seconds = float(latent_seconds)
        self.sampling_weight_mode = str(sampling_weight or "hours")
        self.num_batches_per_epoch = int(num_batches_per_epoch) if num_batches_per_epoch is not None else None
        self.dynamic_even_window = bool(dynamic_even_window)
        self.min_latent_window = max(2, int(min_latent_window))
        if self.min_latent_window % 2:
            self.min_latent_window += 1
        self.max_total_latent_frames = (
            int(max_total_latent_frames)
            if max_total_latent_frames is not None and int(max_total_latent_frames) > 0
            else None
        )
        if temporal_buckets:
            logger.warning("pretrain_sc ignores temporal_buckets when dynamic_even_window=True")
        if batch_size_by_window:
            logger.warning("pretrain_sc ignores batch_size_by_window; use training.batch_size/gradient_accumulation_steps")
        super().__init__(**kwargs)

    @staticmethod
    def _stable_unit_hash(seed: int, text: str) -> float:
        h = hashlib.blake2b(f"{seed}|{text}".encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, "big") / float(1 << 64)

    @staticmethod
    def _source_name_from_path(path: str) -> str:
        return _strip_remote_scheme(str(path).rstrip("/")).rsplit("/", 1)[-1]

    @staticmethod
    def _source_match_patterns(source: Dict[str, Any]) -> List[str]:
        patterns = []
        name = source.get("dataset") or source.get("name")
        if name:
            patterns.append(str(name))
        raw_source = source.get("source")
        if raw_source:
            text = _normalize_remote_path(str(raw_source).rstrip("/"))
            core = _strip_remote_scheme(text)
            patterns.extend([text, f"*{core}*"])
            patterns.append(core.rsplit("/", 1)[-1])
        for key in ("match", "matches", "match_patterns"):
            value = source.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                patterns.append(value)
            else:
                patterns.extend(str(v) for v in value)
        return [p for p in patterns if p]

    def _source_ratio_map(self) -> Dict[str, float]:
        ratios: Dict[str, float] = {}
        for source in self.stereo_latent_sources:
            source_name = str(
                source.get("dataset")
                or source.get("name")
                or self._source_name_from_path(source.get("source", ""))
            )
            value = source.get("ratio", source.get("target_hours"))
            if value is None:
                continue
            try:
                ratio = float(value)
            except (TypeError, ValueError):
                continue
            if ratio > 0:
                ratios[source_name] = ratio
        return ratios

    def _source_key_for_episode(self, ep: Dict[str, Any]) -> str:
        texts = [
            str(ep.get("dataset_name") or ""),
            str(ep.get("episode_path") or ""),
            str(ep.get("latent_root") or ""),
        ]
        for source in self.stereo_latent_sources:
            source_name = str(
                source.get("dataset")
                or source.get("name")
                or self._source_name_from_path(source.get("source", ""))
            )
            patterns = self._source_match_patterns(source)
            if any(self._name_matches(text, pattern) for text in texts for pattern in patterns):
                return source_name
        return str(ep.get("dataset_name") or "unknown")

    def _episode_rank_text(self, ep: Dict[str, Any]) -> str:
        return str(
            ep.get("episode_storage_key")
            or ep.get("episode_path")
            or ep.get("latent_path")
            or ep.get("episode_name")
            or ""
        )

    def _lookup_episode_metadata(self, ep: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        phase_dirs = ep.get("sc_valid_phase_dirs")
        frames = ep.get("sc_total_latent_frames")
        if phase_dirs and frames is not None:
            seconds = ep.get("sc_duration_sec")
            return {
                "phase_dirs": list(phase_dirs),
                "lang_files": [],
                "latent_scanned": True,
                "lang_scanned": False,
                "total_latent_frames": int(frames),
                "duration_sec": float(seconds) if seconds is not None else float(frames) * self.latent_seconds,
            }
        return super()._lookup_episode_metadata(ep)

    def _episode_length_metadata(self, ep: Dict[str, Any]) -> Tuple[Optional[int], Optional[float]]:
        frames = ep.get("sc_total_latent_frames")
        seconds = ep.get("sc_duration_sec")
        if frames is not None:
            if seconds is None:
                seconds = float(frames) * self.latent_seconds
            return int(frames), float(seconds)

        metadata = self._lookup_episode_metadata(ep)
        if metadata is None:
            return None, None
        frames = metadata.get("total_latent_frames")
        seconds = metadata.get("duration_sec")
        if frames is None and metadata.get("phase_latent_frames"):
            frames = sum(
                int(item.get("num_latents") or 0)
                for item in metadata["phase_latent_frames"]
                if isinstance(item, dict)
            )
        if seconds is None and frames is not None:
            seconds = float(frames) * self.latent_seconds
        return (int(frames) if frames is not None else None, float(seconds) if seconds is not None else None)

    def _dynamic_even_window_for_length(self, num_latents: int) -> int:
        window = min(int(num_latents), int(self.max_latent_window))
        if window % 2:
            window -= 1
        if window < self.min_latent_window:
            return 0
        return window

    def _organize_episodes(self, episodes: List[Dict[str, Any]]):
        by_source: Dict[str, List[Dict[str, Any]]] = {}
        for ep in episodes:
            source_key = self._source_key_for_episode(ep)
            cur = dict(ep)
            cur["sc_source"] = source_key
            cur["sc_hash_rank"] = self._stable_unit_hash(self.subset_seed, self._episode_rank_text(cur))
            by_source.setdefault(source_key, []).append(cur)

        selected: List[Dict[str, Any]] = []
        for source_key, source_eps in sorted(by_source.items()):
            ordered = sorted(source_eps, key=lambda item: (item["sc_hash_rank"], self._episode_rank_text(item)))
            keep = max(1, int(math.ceil(len(ordered) * self.subset_fraction)))
            selected.extend(ordered[:keep])

        total_frames = 0
        total_seconds = 0.0
        missing_lengths = 0
        too_short = 0
        too_long = 0
        expanded: List[Dict[str, Any]] = []
        for ep in selected:
            num_latents, seconds = self._episode_length_metadata(ep)
            if num_latents is None:
                missing_lengths += 1
            else:
                total_frames += int(num_latents)
            if seconds is not None:
                total_seconds += float(seconds)

            if (
                self.max_total_latent_frames is not None
                and num_latents is not None
                and int(num_latents) > self.max_total_latent_frames
            ):
                too_long += 1
                continue

            metadata_window_size = None
            if self.dynamic_even_window and num_latents is not None:
                metadata_window_size = self._dynamic_even_window_for_length(num_latents)
                if metadata_window_size <= 0:
                    too_short += 1
                    continue
            cur = dict(ep)
            cur["sc_num_latents"] = num_latents
            cur["sc_duration_sec"] = seconds
            cur["sc_base_weight"] = self._base_episode_weight(num_latents, seconds)
            if metadata_window_size is not None:
                cur["window_size"] = metadata_window_size
            expanded.append(cur)

        self._apply_source_ratios(expanded)

        self.episode_files = expanded
        self._rebuild_task_index()
        self._sc_total_seconds = total_seconds
        self._sc_total_latent_frames = total_frames
        self._sc_missing_lengths = missing_lengths
        self._sc_too_short = too_short
        self._sc_too_long = too_long

    def _base_episode_weight(self, num_latents: Optional[int], seconds: Optional[float]) -> float:
        if self.sampling_weight_mode in {"hours", "duration", "seconds"}:
            if seconds is not None and seconds > 0:
                return float(seconds)
            if num_latents is not None and num_latents > 0:
                return float(num_latents) * self.latent_seconds
        if self.sampling_weight_mode in {"latents", "frames"}:
            if num_latents is not None and num_latents > 0:
                return float(num_latents)
        return 1.0

    def _apply_source_ratios(self, episodes: List[Dict[str, Any]]) -> None:
        ratio_map = self._source_ratio_map()
        by_source: Dict[str, List[Dict[str, Any]]] = {}
        for ep in episodes:
            by_source.setdefault(str(ep.get("sc_source") or "unknown"), []).append(ep)

        for source, source_eps in by_source.items():
            base_sum = sum(float(ep.get("sc_base_weight") or 1.0) for ep in source_eps)
            if base_sum <= 0:
                base_sum = float(len(source_eps))
            target_source_weight = ratio_map.get(source)
            if target_source_weight is None:
                target_source_weight = base_sum
            for ep in source_eps:
                base = float(ep.get("sc_base_weight") or 1.0)
                ep["sc_sample_weight"] = base * float(target_source_weight) / base_sum

    def _episode_for_index(self, idx: int, attempt: int = 0) -> Optional[Dict[str, Any]]:
        if not self.episode_files or self.total_episodes <= 0:
            return None

        base_idx = idx % self.total_episodes
        base_ep = self.episode_files[base_idx]
        if self._episode_identity(base_ep) not in self._dropped_episode_ids:
            return base_ep

        bucket_key = str(base_ep.get("bucket_key") or "unknown")
        candidate_indices = self._bucket_to_episode_indices.get(bucket_key) or list(range(self.total_episodes))
        start = 0
        if base_idx in candidate_indices:
            start = candidate_indices.index(base_idx)
        for offset in range(1, len(candidate_indices) + 1):
            candidate = self.episode_files[candidate_indices[(start + offset) % len(candidate_indices)]]
            if self._episode_identity(candidate) not in self._dropped_episode_ids:
                return candidate
        return base_ep

    def _assign_resolution_buckets(self) -> None:
        super()._assign_resolution_buckets()

    def get_sample_weight(self, idx: int) -> float:
        if not self.episode_files:
            return 1.0
        ep = self.episode_files[idx % self.total_episodes]
        return max(0.0, float(ep.get("sc_sample_weight") or ep.get("sc_base_weight") or 1.0))

    def _window_size_for_episode(self, ep: Dict[str, Any], idx: int) -> int:
        return int(ep.get("window_size") or self.max_latent_window)

    def _effective_window_size_for_episode(
        self,
        ep: Dict[str, Any],
        idx: int,
        total_latents: int,
        requested_window_size: int,
    ) -> int:
        if not self.dynamic_even_window:
            return int(requested_window_size)
        return self._dynamic_even_window_for_length(total_latents)

    def _log_episode_summary(self):
        super()._log_episode_summary()
        if not self.episode_files:
            return
        bucket_counts: Dict[str, int] = {}
        source_counts: Dict[str, int] = {}
        source_weights: Dict[str, float] = {}
        for ep in self.episode_files:
            bucket_counts[str(ep.get("bucket_key") or "unknown")] = bucket_counts.get(str(ep.get("bucket_key") or "unknown"), 0) + 1
            source = str(ep.get("sc_source") or ep.get("dataset_name") or "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
            source_weights[source] = source_weights.get(source, 0.0) + float(ep.get("sc_sample_weight") or 0.0)
        logger.info(
            "PretrainSCDataset: subset_fraction=%.4f seed=%d episodes=%d estimated_hours=%.2f "
            "latent_frames=%d missing_length_metadata=%d too_short=%d too_long=%d "
            "max_total_latent_frames=%s dynamic_even_window=%s min_latent_window=%d "
            "max_latent_window=%d buckets=%s sources=%s source_weights=%s",
            self.subset_fraction,
            self.subset_seed,
            self.total_episodes,
            float(getattr(self, "_sc_total_seconds", 0.0)) / 3600.0,
            int(getattr(self, "_sc_total_latent_frames", 0)),
            int(getattr(self, "_sc_missing_lengths", 0)),
            int(getattr(self, "_sc_too_short", 0)),
            int(getattr(self, "_sc_too_long", 0)),
            self.max_total_latent_frames,
            self.dynamic_even_window,
            self.min_latent_window,
            int(self.max_latent_window),
            dict(sorted(bucket_counts.items())),
            dict(sorted(source_counts.items())),
            {k: round(v, 6) for k, v in sorted(source_weights.items())},
        )
