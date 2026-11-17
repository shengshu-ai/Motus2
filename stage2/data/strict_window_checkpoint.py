"""CPU-only integrity check used by the strict-run supervisor."""
import argparse
import hashlib
import json
from pathlib import Path


def validate(checkpoint, expected_step=None, world_size=32, batch_size=4):
    root = Path(checkpoint)
    marker = json.loads((root / 'checkpoint_complete.json').read_text())
    step = int(marker['global_step'])
    if expected_step is not None and step != int(expected_step):
        raise ValueError('Checkpoint step differs from requested step')
    if int(marker['world_size']) != int(world_size):
        raise ValueError('Checkpoint world size differs')
    for name, size in marker['files'].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Checkpoint member escapes root')
        if not path.is_file() or path.stat().st_size != int(size):
            raise ValueError(f'Incomplete checkpoint member: {name}')
    binding = marker['strict_window']
    if binding['schema'] != 'motus-strict-window-loader-v1' or binding['file'] != 'strict_window_state.json':
        raise ValueError('Unknown strict checkpoint binding')
    if binding['file'] not in marker['files'] or int(binding['committed_step']) != step:
        raise ValueError('Ledger is outside checkpoint manifest or step differs')
    path = root / binding['file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != binding['sha256']:
        raise ValueError('Ledger content hash differs')
    state = json.loads(path.read_text())
    if state['schema'] != binding['schema'] or int(state['committed_step']) != step:
        raise ValueError('Ledger schema/commit differs')
    ledger = state['ledger']
    if int(ledger['committed_step']) != step:
        raise ValueError('Coordinator commit differs')
    identity = ledger['binding']
    if int(identity['world_size']) != int(world_size) or int(identity['batch_size']) != int(batch_size):
        raise ValueError('Coordinator topology differs')
    if ledger.get('requests'):
        raise ValueError('Complete checkpoint has uncommitted requests')
    return dict(passed=True, global_step=step, world_size=int(world_size), batch_size=int(batch_size),
                ledger_sha256=binding['sha256'], validation_scope='CPU manifest and ledger binding; trainer verifies full catalog on load')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--expected-step', type=int)
    parser.add_argument('--world-size', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(validate(args.checkpoint, args.expected_step, args.world_size, args.batch_size), sort_keys=True))
