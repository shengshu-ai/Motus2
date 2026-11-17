"""Crop only fully supervised latent blocks inside their selected language event."""


def choose_event_window(segments, total_latents, window, phase_start, frame_stride, rng):
    source_frames = max(s['end_frame_exclusive'] for s in segments)
    choices = []
    for event in segments:
        eligible = []
        for index in range(total_latents):
            first = phase_start if index == 0 else phase_start + 1 + (index - 1) * 4 * frame_stride
            end = phase_start + 1 if index == 0 else first + 4 * frame_stride
            first, end = min(first, source_frames - 1), min(end, source_frames)
            if event['start_frame'] <= first and end <= event['end_frame_exclusive']:
                eligible.append(index)
        if eligible:
            # Segments and latent blocks are contiguous intervals.
            assert eligible == list(range(eligible[0], eligible[-1] + 1))
            choices.append((event, eligible[0], eligible[-1]))
    if not choices:
        raise ValueError('No complete latent block fits a VPP2 event in this phase')
    event, first, last = rng.choice(choices)
    start = rng.randint(first, max(first, last - window + 1))
    count = min(window, last - start + 1)
    return event, start, count
