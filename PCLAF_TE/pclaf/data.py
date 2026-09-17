"""Training subsets, training-only standardization, and trajectory windows."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Scalers:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray

    @classmethod
    def fit(cls, x_rows, y_endpoints):
        x = np.asarray(x_rows, dtype=np.float64)
        y = np.asarray(y_endpoints, dtype=np.float64).reshape(-1, 1)
        return cls(x.mean(0), np.maximum(x.std(0), 1e-8),
                   y.mean(0), np.maximum(y.std(0), 1e-8))

    def transform_x(self, x):
        return np.asarray((x - self.x_mean) / self.x_std, dtype=np.float32)

    def transform_y(self, y):
        return np.asarray((y - self.y_mean) / self.y_std, dtype=np.float32)

    def inverse_x(self, x):
        return np.asarray(x) * self.x_std + self.x_mean

    def inverse_y(self, y):
        return np.asarray(y) * self.y_std + self.y_mean

    def to_dict(self):
        return {name: getattr(self, name).tolist()
                for name in ('x_mean', 'x_std', 'y_mean', 'y_std')}

    @classmethod
    def from_dict(cls, values):
        return cls(**{name: np.asarray(values[name], dtype=np.float64)
                      for name in ('x_mean', 'x_std', 'y_mean', 'y_std')})


def load_data(directory):
    directory = Path(directory)
    arrays = []
    for name in ('train', 'test'):
        with np.load(str(directory / (name + '.npz')), allow_pickle=False) as z:
            value = {key: z[key] for key in z.files}
        if any(not np.isfinite(a).all() for a in value.values()):
            raise ValueError('Nonfinite values in ' + name)
        arrays.append(value)
    train, test = arrays
    expected = {'x': (275, 20, 23), 'y': (275, 1),
                'condition': (275,), 'run': (275,), 'end_index': (275,)}
    expected_test = {'process': (120, 960, 23), 'quality': (120, 960, 1),
                     'condition': (120,), 'run': (120,)}
    for value, shapes in ((train, expected), (test, expected_test)):
        if set(value) != set(shapes) or any(value[k].shape != s for k, s in shapes.items()):
            raise ValueError('Unexpected TE data schema')
    if set(zip(test['run'], test['condition'])) != {
            (r, c) for r in range(31, 61) for c in range(4)}:
        raise ValueError('Expected testing runs 31-60 in conditions D00-D03')
    return train, test


def subset_indices(train, scenario, labels, seed):
    """Sample within run/condition groups, independently of target values."""
    if scenario not in ('s1', 's2'):
        raise ValueError('scenario must be s1 or s2')
    if scenario == 's1' and labels not in (10, 20, 55, 110, 275):
        raise ValueError('s1 labels must be 10, 20, 55, 110, or 275')
    if scenario == 's2' and (labels < 10 or labels > 125 or labels % 5):
        raise ValueError('s2 labels must be a multiple of 5 from 10 to 125')
    ten = {1: (0, 1), 2: (0, 2), 3: (0, 3), 4: (0, 3), 5: (1, 2)}
    selected = []
    for run in range(1, 6):
        for condition in (range(4) if scenario == 's1' else (0,)):
            pool = np.flatnonzero((train['run'] == run) &
                                  (train['condition'] == condition))
            rng = np.random.RandomState((seed + 710000 + run * 1009
                                         + condition * 9176) % 2**32)
            if scenario == 's2':
                count = labels // 5
            elif labels == 10:
                count = int(condition in ten[run])
            elif labels == 20:
                count = 1
            else:
                count = {55: (5, 2), 110: (10, 4), 275: (25, 10)}[labels][int(condition != 0)]
            if count > len(pool):
                raise ValueError('Insufficient candidates in a training group')
            selected.extend(rng.permutation(pool)[:count].tolist())
    indices = np.sort(np.asarray(selected, dtype=np.int64))
    if len(indices) != labels:
        raise ValueError('Training subset size mismatch')
    return indices


def training_data(train, scenario, labels, seed):
    indices = subset_indices(train, scenario, labels, seed)
    x, y = train['x'][indices], train['y'][indices]
    scaler = Scalers.fit(x.reshape(-1, 23), y)
    real = dict(x=scaler.transform_x(x), y=scaler.transform_y(y),
                condition=train['condition'][indices].copy())
    return real, scaler, indices


def trajectory_batches(process, scaler, batch_size):
    """Windows stay within a trajectory; endpoint indices are zero-based."""
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    x = scaler.transform_x(process)
    for start in range(19, len(x), batch_size):
        endpoints = np.arange(start, min(start + batch_size, len(x)))
        yield endpoints, np.stack([x[t-19:t+1] for t in endpoints])
