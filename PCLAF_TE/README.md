# PCLAF: Tennessee Eastman experiments

Train PCLAF, VAE, GAN, and Raw comparisons from the included TE data, then
export MAE, RMSE, numerical tables, and boxplots. Each input is a window of
20 observations with 23 measured variables; the output is one XMEAS(35)
quality value at the window endpoint.

## Run

Install Python with PyTorch, NumPy, and Matplotlib. A CUDA GPU is supported;
CPU execution is also available. Install the dependencies from this directory:

```bash
python -m pip install -r requirements.txt
python run.py --scenario both
```

The default run uses the same 10 seeds for all methods:
`9701, 9703, 9707, 9719, 9721, 9733, 9739, 9743, 9749, 9767`.
Predictors train on CPU. Generators and evaluation use CUDA when available;
`--device cpu` runs everything on CPU. Use `--predictor-device` to change the
predictor training device. Each run records the device and library versions.
GPU convolutions use deterministic cuDNN algorithms, with cuDNN benchmarking
disabled.

To run a single shared seed or select a device:

```bash
python run.py --scenario s1 --seeds 9701 --device cpu --output outputs/s1_seed9701
python run.py --scenario s2 --seeds 9701 --device cuda:0 --output outputs/s2_seed9701
```

Use a new output directory for each invocation. Every model is trained from
scratch. The program records its resolved settings, data hashes, software
versions, training subset, training logs, and newly trained checkpoints in
that directory.

## Comparisons and settings

Scenario 1 trains on D00–D03 and evaluates each condition in Testing runs
31–60. Scenario 2 trains on D00 and evaluates generalization to D01–D03;
its tables also include D00. Both scenarios use 10 labeled pairs by default
and generate 64 synthetic pairs per real pair, with counts preserved by
condition. Each generated bank is shared across its downstream predictors.

| Predictor | Architecture settings | Learning rate | S1 updates | S2 updates | Predictor physics weight |
|---|---|---:|---:|---:|---:|
| CNN | 3 convolution blocks, dropout 0.6 | 0.003 | 330 | — | 0 |
| LSTM | 2 bidirectional layers, hidden size 64, dropout 0.2, attention | 0.001 | 90 | 90 | 0 |
| Transformer | 2 layers, model size 12, 4 heads, dropout 0.1 | 0.003 | 80 | — | 0 |
| PI-LSTM | Same architecture as LSTM | 0.001 | 210 | 90 | 1 |

Within each predictor, Raw and all augmentation methods share the architecture,
learning rate, number of updates, initialization, real minibatch sequence, and
real-pass dropout sequence. Batches contain 32 samples, balanced across the
training conditions. Generated samples have a separate random stream and an
additional loss weight of 0.25. Training returns the final scheduled update.
Predictor inputs contain the 23 measured variables. LSTM, PI-LSTM, and
Transformer append three relative position features inside the network.

The predictor objective is
`MSE(real) + rho * physics(real) + 0.25 * [MSE(synthetic) + rho * physics(synthetic)]`;
the synthetic term is absent for Raw. `rho` is the predictor physics weight
in the table. It is separate from physical regularization and latent guidance
inside the PCLAF generator.

PCLAF trains a conditional joint VAE for 250 epochs and a latent DDPM for
1,500 updates. Its decoder uses the physical residual in `pclaf/physics.py`.
Sampling uses 100 diffusion steps and up to 10 latent guidance steps, with
backtracking, a trust radius of 0.5, and tolerance `1e-8`. The residual uses
the five product compositions D–H and five process measurements, with the
projection and coefficients stated directly in that module. This VLE proxy
assumes small stripper accumulation and uses five-point means, with a
two-sample lag for the additional process measurements. VAE and GAN use
convolutional generators trained for 300 epochs. Both baselines use min/max
scaling fitted to their training pairs, mapped to `[0,1]`, and Tanh decoder
outputs. Their generated values are mapped back through the
same fitted transform. The GAN objective uses binary cross-entropy only.

Export the complete default settings to edit architecture, learning rates,
training updates, label count, generation ratio, and generator parameters:

```bash
python run.py --write-config settings.json
python run.py --config settings.json --scenario both --output outputs/custom
```

The same predictor settings apply to every data method. A JSON file can also
contain just the fields to override. Scenario 1 supports label counts
10, 20, 55, 110, and 275; Scenario 2 supports multiples of 5 from 10 to 125.
Use a count supported by both when running both scenarios together.

## Data and metrics

The included candidate pool contains 275 labeled pairs from Training runs
1–5. Standardization is fitted to the selected training pairs. Training
windows have length 20 and stride 20. Testing windows have length 20 and
stride 1, formed separately within each 960-point trajectory, giving
941 endpoints per condition and run. See [data/te/README.md](data/te/README.md)
for the dataset citation, variable order, units, fields, and source terms.

MAE and RMSE are exported on two scales:

- `mole_percentage_points`: errors in the original XMEAS(35) units.
- `range_normalized`: errors divided by the full 960-point target range of
  the same testing trajectory. This dimensionless scale is applied only
  when reporting metrics.

For each seed, metrics are first calculated for each condition and testing
run. The Scenario 1 aggregate averages all four conditions equally within
each run. The Scenario 2 aggregate averages D01–D03 equally within each run.

The `reports/` directory contains:

- `preview.png`, `preview.pdf`, and `preview.svg`: the default RMSE preview
  for Scenario 1, PI-LSTM, D01, using the first requested seed (9701 by default).
  This panel is exported when the selected run includes that comparison.
- `metrics_by_run.csv`: per-seed, per-condition and aggregate run metrics.
- `metrics_by_seed.csv`: means and sample standard deviations across the
  30 testing runs for each seed.
- `summary.csv`: means and sample standard deviations across seed-level
  means. The standard deviation is blank when only one seed is run.
- PNG, PDF, and SVG boxplots, separately for each seed and reporting scale.
  Each box contains the same 30 testing runs; Scenario 2 uses the per-run
  fault-condition average. Boxes show quartiles and medians, whiskers use
  1.5 times the interquartile range, and diamonds show means. Within each
  condition, all predictors share the same vertical scale.
