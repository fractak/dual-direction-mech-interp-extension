"""Resolve single-token controls on the exact rendered prompt, without a model."""
POSITION_CONTROL_MODES = {
    'position_final_fixed': 'final_prefill',
    'position_previous_fixed': 'previous_prefill',
    'position_two_back_fixed': 'two_back_prefill',
    'position_user_last_fixed': 'last_user_content',
}

def locate_positions(rendered, prompt, ids, offsets):
    if not prompt or rendered.count(prompt) != 1:
        raise ValueError('User content must occur exactly once in the rendered prompt')
    if len(ids) != len(offsets) or len(ids) < 4:
        raise ValueError('Token/offset lengths differ or sequence is too short')
    start = rendered.index(prompt)
    end = start + len(prompt)
    content = []
    previous = 0
    for i, pair in enumerate(offsets):
        if len(pair) != 2:
            raise ValueError('Malformed token offset')
        a, b = pair
        if not (0 <= a <= b <= len(rendered)):
            raise ValueError('Token offset outside rendered text')
        if b > a:
            if a < previous:
                raise ValueError('Nonmonotonic token offsets')
            previous = a
        if b > start and a < end and b > a:
            if a < start or b > end:
                raise ValueError('A token straddles the user-content boundary')
            content.append(i)
    if not content or offsets[content[-1]][1] != end:
        raise ValueError('Last user-content token is missing or input was truncated')
    positions = {'final_prefill':len(ids)-1, 'previous_prefill':len(ids)-2,
                 'two_back_prefill':len(ids)-3, 'last_user_content':content[-1]}
    if len(set(positions.values())) != 4:
        raise ValueError('Position controls are not four distinct token positions')
    return positions
