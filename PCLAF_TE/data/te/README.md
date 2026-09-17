# Tennessee Eastman data

These files contain the measured data subset used by this package from Rieth,
Amsel, Tran, and Cook (2017), *Additional Tennessee Eastman Process Simulation
Data for Anomaly Detection Evaluation*, Harvard Dataverse:
[doi:10.7910/DVN/6C3JR1](https://doi.org/10.7910/DVN/6C3JR1).
Samples are recorded every three minutes. Training and Testing are distinct
partitions of the public dataset; their simulation run identifiers are local
to each partition.

The dataset's published description includes a public-domain dedication and
permits copying, modification, and distribution for lawful purposes. It also
disclaims warranties and liability and prohibits implying the authors' or
Pacific Science & Engineering Group's endorsement. See the
[original dataset terms](https://dataverse.harvard.edu/api/datasets/:persistentId/versions/1.0/customlicense?persistentId=doi:10.7910/DVN/6C3JR1).
These terms apply to the source data. Cite the dataset when using these
measurements.

## Variables and storage

The 23 input columns, in order, are XMEAS(23–34), XMEAS(36–41), followed by
XMEAS(4), XMEAS(11), XMEAS(13), XMEAS(14), and XMEAS(18). The target is
XMEAS(35), expressed in mole percent. Stored input measurements retain their
original values and source units.

Both archives load with `numpy.load(path, allow_pickle=False)`. Measurements
are `float64`; identifiers and row indices are `int64`. `condition` is the
public `faultNumber` (0 denotes normal operation), and `run` is the public
`simulationRun`. These identifiers describe groups; predictor inputs are the
23 measured columns listed above. Exact column names, array shapes,
source-file identities, and archive SHA256 values are recorded in `metadata.json`.

## Training: `train.npz`

- `x`: 275 process windows, shaped `(275, 20, 23)`.
- `y`: one measured endpoint target per window, shaped `(275, 1)`.
- `condition`, `run`, `end_index`: one identifier or zero-based trajectory
  endpoint row per window, each shaped `(275,)`.

The candidate pool uses Training runs 1–5 and conditions 0–3. Windows have
length 20 and stride 20. Each run contributes 25 normal-operation endpoints
from the first 500 rows and 10 endpoints per fault condition from the first
200 rows: 55 pairs per run, 275 in total. The runtime selects its configured
labeled subset from this pool. The measured endpoint quality is stored as
the target; input windows contain process measurements.

## Testing: `test.npz`

- `process`: 120 observed trajectories, shaped `(120, 960, 23)`.
- `quality`: their full target trajectories, shaped `(120, 960, 1)`.
- `condition`, `run`: one identifier per trajectory, each shaped `(120,)`.

The test set uses Testing runs 31–60 and conditions 0–3, ordered by increasing
run and then condition. Each trajectory contains source sample rows 1–960.
Create length-20, stride-1 windows independently within each trajectory.
Zero-based rows 19–959 are the 941 quality endpoints per trajectory, giving
112,920 endpoints overall.

All 960 quality values are retained so the optional dimensionless reporting
scale can use each trajectory's full target range, `max(quality)-min(quality)`.
That range is applied only when reporting errors. Model scalers are fitted
to the selected training pairs.
