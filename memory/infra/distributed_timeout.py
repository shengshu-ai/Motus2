"""Keep DeepSpeed's data-parallel groups on the configured training timeout."""

from datetime import timedelta

from torch.distributed.distributed_c10d import _set_pg_timeout


def configure_deepspeed_group_timeouts(engine, timeout: timedelta) -> None:
    # InitProcessGroupKwargs only configures WORLD. DeepSpeed clones that group
    # during prepare(), without passing a timeout to new_group(). PyTorch then
    # uses its backend default (600s for NCCL), not WORLD's configured timeout.
    optimizer = getattr(engine, "optimizer", None)
    groups = (
        getattr(engine, "data_parallel_group", None),
        getattr(engine, "seq_data_parallel_group", None),
        getattr(optimizer, "dp_process_group", None),
    )
    seen = set()
    for group in groups:
        if group is not None and id(group) not in seen:
            _set_pg_timeout(timeout, group)
            seen.add(id(group))
