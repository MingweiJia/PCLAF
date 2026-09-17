"""Shared predictor settings for Raw and every augmentation method."""
import copy

from .generators import DEFAULT_PCLAF_CONFIG, BASELINE_CONFIGS


REPEAT_SEEDS = [9701, 9703, 9707, 9719, 9721, 9733, 9739, 9743, 9749, 9767]
DEFAULT_SEEDS = REPEAT_SEEDS


def default_config():
    shared = dict(batch_size=32, synthetic_weight=.25, gradient_clip=5.,
                  eval_interval=10, weight_decay=0.)
    predictors = {
        'cnn': dict(model=dict(dropout=.6), learning_rate=.003, rho=0.),
        'lstm': dict(model=dict(hidden_dim=64, num_layers=2, dropout=.2),
                     learning_rate=.001, rho=0.),
        'transformer': dict(model=dict(d_model=12, nhead=4, num_layers=2, dropout=.1),
                            learning_rate=.003, rho=0.),
        'pilstm': dict(model=dict(hidden_dim=64, num_layers=2, dropout=.2),
                       learning_rate=.001, rho=1.),
    }
    for value in predictors.values():
        value.update(shared)
    return dict(
        labels=10, synthetic_ratio=64, methods=['raw', 'pclaf', 'vae', 'gan'],
        predictors=predictors,
        scenarios={
            's1': dict(models=['cnn', 'lstm', 'transformer', 'pilstm'],
                       updates=dict(cnn=330, lstm=90, transformer=80, pilstm=210)),
            's2': dict(models=['lstm', 'pilstm'], updates=dict(lstm=90, pilstm=90)),
        },
        generators=dict(pclaf=copy.deepcopy(DEFAULT_PCLAF_CONFIG),
                        vae=copy.deepcopy(BASELINE_CONFIGS['vae']),
                        gan=copy.deepcopy(BASELINE_CONFIGS['gan'])),
    )


def merge_config(base, changes, path='config'):
    """Merge editable settings while rejecting misspelled parameter names."""
    if not isinstance(changes, dict):
        raise ValueError(path + ' must be an object')
    result = copy.deepcopy(base)
    for key, value in changes.items():
        if key not in result:
            raise ValueError('Unknown option: ' + path + '.' + key)
        if isinstance(result[key], dict):
            result[key] = merge_config(result[key], value, path + '.' + key)
        else:
            result[key] = value
    return result
