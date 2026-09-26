"""Shared per-layer variance targets for the residual piece-token Transformer."""

RESIDUAL_BRANCH_TARGET_VARIANCE = 0.125
TRANSFORMER_INPUT_TARGET_VARIANCE = 1.0
TRANSFORMER_QUERY_KEY_TARGET_VARIANCE = 0.0004
TRANSFORMER_VALUE_TARGET_VARIANCE = 0.04


def transformer_parameter_target_variance(key):
    """Return a target variance coefficient C for ``variance = C / fan_in``."""
    if key == 'W1':
        return TRANSFORMER_INPUT_TARGET_VARIANCE
    if key.startswith(('WQ','WK')):
        return TRANSFORMER_QUERY_KEY_TARGET_VARIANCE
    if key.startswith('WV'):
        return TRANSFORMER_VALUE_TARGET_VARIANCE
    if key.startswith('W') and key[1:].isdigit():
        return RESIDUAL_BRANCH_TARGET_VARIANCE
    return None
