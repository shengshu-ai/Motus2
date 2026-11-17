"""Sampler that concatenates independent shuffles into one epoch.

Repeating permutations reduces DataLoader epoch-boundary synchronization
while preserving the random-sampling distribution. ``repeat=1`` produces a
single permutation of the dataset.
"""

from typing import Iterator, Optional, Sized

import torch
from torch.utils.data import Sampler


class RepeatedRandomSampler(Sampler[int]):
    """`repeat` independent permutations of `data_source`, yielded end to end.

    With ``repeat == 1`` this is exactly ``RandomSampler(replacement=False)``.
    """

    def __init__(
        self,
        data_source: Sized,
        repeat: int,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        repeat = int(repeat)
        if repeat < 1:
            raise ValueError(f"repeat must be >= 1, got {repeat}")
        self.data_source = data_source
        self.repeat = repeat
        self.generator = generator

    def __len__(self) -> int:
        return len(self.data_source) * self.repeat

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        generator = self.generator
        if generator is None:
            # Same seeding contract as torch's RandomSampler with generator=None:
            # draw one fresh seed from the global RNG per iteration.
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        for _ in range(self.repeat):
            yield from torch.randperm(n, generator=generator).tolist()
