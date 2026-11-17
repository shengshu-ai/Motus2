"""Fixed evaluation pools with explicit exposure labels and spatial buckets."""
import re

from torch.utils.data import Subset


class EvaluationSubset(Subset):
    def get_bucket_key(self, index):
        return self.dataset.get_bucket_key(self.indices[index])


def evaluation_subsets(dataset, groups):
    """Each declared group gets its own loader, so a short eval cannot omit it."""
    result, claimed = {}, set()
    for name, spec in groups.items():
        if not re.fullmatch(r'[A-Za-z0-9_]+', name):
            raise ValueError('Evaluation group names must be safe metric labels')
        sources = set(spec['sources'])
        indices = [i for i, ep in enumerate(dataset.episode_files)
                   if dataset._sampling_source_identity(ep) in sources]
        if not indices:
            raise ValueError(f'Empty evaluation group: {name}')
        if claimed.intersection(indices):
            raise ValueError(f'Overlapping evaluation groups: {name}')
        claimed.update(indices)
        result[name] = EvaluationSubset(dataset, indices)
    return result


def prefix_evaluation_metrics(name, values):
    """Preserve per-metric counts, sums and images across group aggregation."""
    prefixed = {}
    for key, value in values.items():
        if key == '_sum_keys':
            prefixed[key] = [f'{name}/{tag}' for tag in value]
        elif key.startswith('_count/'):
            prefixed[f'_count/{name}/{key[7:]}'] = value
        elif key.startswith('visualization_'):
            prefixed[f'visualization_{name}_{key[14:]}'] = value
        else:
            prefixed[f'{name}/{key}'] = value
    return prefixed
