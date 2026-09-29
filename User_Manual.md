# MLdpit — User Manual

**Deep learning temporal downscaling of precipitation: hourly to 10-minute**

Version 1.0.0

---

## Contents

1. [Introduction](#1-introduction)
2. [Software overview](#2-software-overview)
3. [Scientific background](#3-scientific-background)
4. [System requirements](#4-system-requirements)
5. [Installation](#5-installation)
6. [Directory structure](#6-directory-structure)
7. [Input data](#7-input-data)
8. [Training workflow](#8-training-workflow)
9. [Inference workflow](#9-inference-workflow)
10. [Example usage](#10-example-usage)
11. [Output description](#11-output-description)
12. [Citation](#12-citation)

---

## 1. Introduction

Many applications need precipitation at a finer temporal resolution than climate
models commonly archive. Urban drainage design, flash-flood modelling and
erosion studies respond to rainfall intensity over minutes, while regional
climate simulations are typically stored as hourly accumulations.

MLdpit converts hourly precipitation into 10-minute precipitation. A
convolutional network predicts how each hour's accumulation is distributed
across the six 10-minute intervals within it, and the hourly total is preserved
exactly by construction.

This manual serves two readers:

- **Users applying the released model** to their own hourly fields: sections 4,
  5, 7, 9, 10 and 11.
- **Reviewers assessing the software** alongside the accompanying manuscript:
  sections 1, 3 and 8.

### A note on the training data

The 1-hour and 10-minute precipitation fields used as training data were generated
within the UDAG project. The corresponding
hourly fields will be available through the CORDEX repository.

The training implementation in this repository is complete: the network, the
mass-conservation constraint, the loss, the data pipeline and the training
configuration used for the published model are all included and documented in
section 8. Without the UDAG targets, a third party cannot repeat the published
training run. A third party can:

- read and audit the full training implementation;
- run it end to end on synthetic data (section 10) or on their own paired
  hourly and sub-hourly data;
- apply the released weights to publicly available hourly fields.

A model trained on a historical period can be applied to hourly output from
future warming scenarios without further training. The accompanying manuscript evaluates
this for future warming scenarios.

---

## 2. Software overview

MLdpit is a Python package with three command-line entry points.

| Command | Purpose |
|---|---|
| `mldpit-preprocess` | Convert yearly NetCDF files into chunked Zarr stores |
| `mldpit-train` | Train the network on paired hourly and 10-minute data |
| `mldpit-infer` | Apply a trained checkpoint to hourly data |

Each accepts `--config file.yaml`. Keys in that file supply the defaults for
the corresponding flags, and a flag given on the command line overrides the
file. 

The components are also importable:

```python
from mldpit.model import LightningSRUNet, softmax_constraint
from mldpit.dataset import LazyXarrayDataset, LazyXarraySourceOnlyDataset
```

### Components

- **`unet.py`** — the network. A four-level U-Net with `DoubleConv` blocks
  (two 3×3 convolutions, each followed by batch normalisation and ReLU),
  max-pooling in the encoder, transposed convolutions in the decoder, skip
  connections at every level, and a 1×1 convolution producing six output
  channels. 
- **`model.py`** — the LightningModule: forward pass, mass-conservation
  constraint, loss and optimiser configuration.
- **`dataset.py`** — lazy Zarr-backed datasets. One yields paired
  hourly/10-minute samples for training; the other yields hourly samples only,
  for inference.
- **`datamodule.py`** — training and validation dataloaders.
- **`preprocess.py`** — NetCDF to Zarr conversion.
- **`train.py`**, **`inference.py`** — the entry points.
- **`config.py`** — YAML configuration handling and filename-template
  expansion, shared by all entry points.

---

## 3. Scientific background

### The problem

Let `P_h` be the precipitation accumulated over hour *h* in a grid cell, and
`p_1 … p_6` the accumulations over the six 10-minute intervals within it. By
definition

```
P_h = p_1 + p_2 + p_3 + p_4 + p_5 + p_6
```

Recovering the six values from their sum is underdetermined: many sub-hourly
distributions give the same total. The network learns which distributions are
physically plausible from the spatial structure of the field and from its
evolution between consecutive hours.

### Inputs

The network receives two channels: the hour to be downscaled and the hour that
follows it. The following hour carries information about the direction of an
event: a system arriving late in the hour leaves a signature in the next.

Both channels are transformed by

```
x = log10(P + ε),    ε = 1e-6
```

Precipitation is strongly skewed, with most cells dry and a few very wet. The
logarithm compresses that range so that the loss is not dominated by the
heaviest cells, and the offset ε keeps the logarithm finite for dry cells.

### The mass-conservation constraint

The network's raw output passes through a constraint before it is used. Writing
`r_1 … r_6` for the raw output channels and `T` for the hourly total in linear
units,

```
w_i = softmax(10^r)_i          weights summing to 1 across the six intervals
p_i = w_i · T                  the constrained prediction
```

Because the softmax weights sum to one, the six predictions sum to `T`
identically. Mass conservation is a property of the architecture; the network
learns only the *shape* of the within-hour distribution.

### Loss and optimisation

The loss is the mean squared error between predicted and target fields in
log10 space. Optimisation uses AdamW at a learning rate of 1e-4, with
`ReduceLROnPlateau` (patience 2, factor 0.1) on validation loss and early
stopping after 5 epochs without improvement.

### Spatial handling

A four-level U-Net halves the resolution four times, which requires spatial
dimensions divisible by 16. The input is reflect-padded up to the next multiple
of 16 and the output cropped back to the original extent. For the 501 × 501
training domain this is a padding of `(5, 6, 5, 6)` to 512 × 512 and a crop of
`[5:506, 5:506]`. Reflection padding avoids introducing an artificial dry
boundary.

---

## 4. System requirements

### Software

| | |
|---|---|
| Python | 3.11 |
| PyTorch | 2.7.1 |
| PyTorch Lightning | 2.5.2 |
| xarray | 2025.7.1 |
| zarr | 3.1.0 |
| numpy | 2.1.3 |
| Others | see `requirements.txt` |

These are the versions with which the published results were produced. The
package generally works with nearby versions; the pins record the
computational environment.

### Hardware

**Inference** runs on CPU. One hour on the 501 × 501 grid takes a few minutes
and needs a few hundred megabytes. A GPU shortens long runs considerably.

**Training** requires GPUs in practice. The published model was trained on
NVIDIA A100 GPUs on the DKRZ Levante system (section 8).

**Memory during inference.** Predictions accumulate in host memory before they
are written, at

```
hours × 6 × 501 × 501 × 4 bytes
```

which is about 6 MB per hour, 4.3 GB per month and 50 GB per year. The
conversion step transiently allocates the same amount again, so a full year
needs more than 100 GB of RAM. On smaller machines, process long periods in
monthly or seasonal chunks with `--period-start` and `--period-end`.

---

## 5. Installation

### With pip and a virtual environment

The package is installed into a Python virtual environment with pip. On HPC
systems, load a Python 3.11 module first if the system default differs.

```bash
git clone https://github.com/midhuna987/MLdpit.git
cd mldpit

python -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The first command installs the exact versions used for the published results.
`--no-deps` on the second prevents pip from re-resolving them.

`requirements.txt` includes `wandb`, which is needed only for optional
Weights & Biases logging during training. Remove that line to install without
it.

### Checking the installation

```bash
python -c "import mldpit; print(mldpit.__version__)"
mldpit-preprocess --help
mldpit-train --help
mldpit-infer --help
```

Section 10 gives a complete end-to-end test that needs no external data.

### Obtaining the weights

The pretrained weights will be archived on Zenodo under CC-BY-4.0, with their
own DOI, after the first round of peer review. Until then they are available to
editors and reviewers on request from the corresponding author.

The Zenodo record will include `best_weight.ckpt` and a `SHA256SUMS` file.
Verify the download before use:

```bash
sha256sum -c SHA256SUMS
```

---

## 6. Directory structure

```
mldpit/
├── README.md
├── User_Manual.md                  this document
├── LICENSE                         MIT
├── CITATION.cff                    citation metadata
├── pyproject.toml                  package definition and entry points
├── requirements.txt                exact versions used for the published results
│
├── src/mldpit/
│   ├── __init__.py
│   ├── config.py                   YAML handling, filename templates
│   ├── unet.py                     network definition
│   ├── model.py                    LightningModule, mass-conservation constraint
│   ├── dataset.py                  lazy Zarr datasets
│   ├── datamodule.py               dataloaders
│   ├── preprocess.py               mldpit-preprocess
│   ├── train.py                    mldpit-train
│   └── inference.py                mldpit-infer
│
├── configs/
│   ├── preprocess_hourly.yaml      NetCDF to Zarr, hourly source fields
│   ├── preprocess_subhourly.yaml   NetCDF to Zarr, 10-minute target fields
│   ├── train.yaml                  configuration of the published training run
│   └── inference.yaml              example inference configuration
│
└── slurm/
    ├── train.sbatch                multi-node GPU training template
    └── inference.sbatch            single-GPU inference template
```

The configuration files record the settings used for the manuscript. Their
directory paths are placeholders and must be set before use.

Running the software creates `checkpoints/`, `logs/` and output directories in
the working directory. These hold run products and are excluded from the
repository by `.gitignore`.

---

## 7. Input data

### Format

Zarr stores, one per year, each holding a variable named `TOT_PREC` (or
`TOT_PREC_10min`; both names are recognised) with dimensions
`(time, rlat, rlon)` and units of `kg m-2` (accumulation over the interval).

### The timestamp convention

Check this before anything else: it is the most common source of error.

Timestamps denote **interval centres**:

| Data | Stamped at | Meaning |
|---|---|---|
| Hourly | `HH:30` | accumulation over `HH:00`–`HH+1:00` |
| 10-minute | `:05, :15, :25, :35, :45, :55` | accumulation over the 10 minutes centred on the stamp |

Check your input:

```python
import xarray as xr
ds = xr.open_zarr("train_source_2024_EXP_1hr.zarr")
print([str(t)[11:16] for t in ds.time.values[:3]])   # expect ['00:30', '01:30', '02:30']
```

If your hourly data are stamped on the hour, shift them before use (see the
 example below). Nothing downstream will warn you: during training no
sample pairs will be found, and during inference the predictions will be
labelled half an hour away from the interval they represent.

### The grid

The released weights were trained on 501 × 501 fields on the rotated-pole grid
of the ICON simulations, with `rlat` and `rlon` as dimension coordinates, `lat`
and `lon` as two-dimensional auxiliary coordinates, and a `rotated_pole`
variable carrying the grid mapping. The network runs on any grid size, but the
learned weights reflect the statistics of this domain and resolution.

### Converting from NetCDF

```bash
mldpit-preprocess \
  --kind hourly \
  --input-dir  /path/to/netcdf \
  --output-dir /path/to/zarr \
  --input-template  "TOT_PREC_{year}010100-{next_year}010100.nc" \
  --output-template "train_source_{year}_{experiment}_1hr.zarr" \
  --experiment EXP \
  --years 2020-2024
```

Use `--kind subhourly` for 10-minute fields. One Zarr store is written per
calendar year. The time chunk defaults to one day: 24 steps for hourly data and
144 for 10-minute data, which suits the access pattern during training.
`configs/preprocess_hourly.yaml` and `configs/preprocess_subhourly.yaml` hold
the settings used for the manuscript.


## 8. Training workflow

Training requires paired hourly and 10-minute data covering the same period on
the same grid, both following the timestamp convention of section 7.

### Configuration

`configs/train.yaml` holds the configuration of the published run. Set
`source_dir`, `target_dir` and the two filename templates to match your data,
then launch:

```bash
mldpit-train --config configs/train.yaml
```

### The published training run

| Setting | Value |
|---|---|
| Training years | 1960–2007, 2012–2017 |
| Validation years | 2018–2019 |
| Not used in training or validation | 2008–2011, 2020–2024 |
| Excluded hour | `2000-01-01T00` |
| Hardware | 12 nodes × 4 NVIDIA A100 GPUs , DKRZ Levante |
| Parallelism | distributed data parallel, synchronised batch normalisation |
| Batch size | 8 per GPU, gradient accumulation 2 |
| Precision | `32-true`, float32 matmul precision `medium` |
| Optimiser | AdamW, learning rate 1e-4 |
| Scheduler | `ReduceLROnPlateau`, patience 2, factor 0.1 |
| Stopping | at most 20 epochs, early stopping patience 5 |
| Checkpoint used | `best_weight.ckpt` (lowest validation loss) |



### Key options

| Flag | Default | Notes |
|---|---|---|
| `--train-years`, `--val-years` | required | ranges and lists, e.g. `"1960-2007,2012-2017"` |
| `--experiment` | empty | fills `{experiment}` in the filename templates |
| `--batch-size` | 4 | per device |
| `--accumulate-grad-batches` | 2 | effective batch is this × batch size × total devices |
| `--devices` | 4 | GPUs per node |
| `--num-nodes` | 1 | |
| `--strategy` | `ddp` | overridden to `auto` on CPU |
| `--accelerator` | `auto` | `gpu` when CUDA is present, otherwise `cpu` |
| `--early-stopping-patience` | 5 | epochs without validation improvement |
| `--exclude-hours` | none | hours to drop, as `YYYY-MM-DDTHH` |
| `--resume` | off | continue from `last.ckpt` |
| `--wandb-project` | none | enables Weights & Biases logging |

On CPU, synchronised batch normalisation and multi-device DDP cannot run, so
`--strategy`, `--devices` and synchronised batch normalisation are set to a
single-process configuration and a notice is printed. This allows a quick
functional test on a login node or laptop.

### Checkpoints

Written to `--checkpoint-dir`:

- `best_weight.ckpt` — lowest validation loss
- `checkpoint_epoch_NNN.ckpt` — one per epoch
- `last.ckpt` — most recent, used by `--resume`

Checkpoints record their hyperparameters, so `load_from_checkpoint` restores
the settings the model was trained with.

### On SLURM

`slurm/train.sbatch` is the template for multi-node training. Before use, set
`--account` to your project and the activation line to the path of the virtual
environment created in section 5, for example
`source .venv/bin/activate`.

The template requests `--ntasks-per-node=4` with `--gpus-per-node=4`: PyTorch
Lightning expects one SLURM task per GPU, and `--devices` in the configuration
must equal the number of tasks per node. The script passes
`--num-nodes "$SLURM_NNODES"`, so the node count is set in one place, the
`#SBATCH --nodes` line.

For a single-GPU test, request `--nodes=1 --ntasks=1 --gpus=1` and run with
`--devices 1 --strategy auto`.

---

## 9. Inference workflow

### Basic use

With explicit input stores:

```bash
mldpit-infer \
  --checkpoint weights/best_weight.ckpt \
  --input /path/to/zarr/train_source_2024_EXP_1hr.zarr \
  --period-start 2024-07-01 --period-end 2024-08-01 \
  --output predictions_2024-07.nc
```

With stores resolved from a filename template:

```bash
mldpit-infer \
  --checkpoint weights/best_weight.ckpt \
  --input-dir /path/to/zarr \
  --input-template "train_source_{year}_{experiment}_1hr.zarr" \
  --experiment EXP \
  --period-start 2020-01-01 --period-end 2020-07-01 \
  --output predictions_2020-H1.nc
```

Or from a configuration file, with any flag overriding it:

```bash
mldpit-infer --config configs/inference.yaml
```

A template may use only `{year}` and `{experiment}`. With a template, one store
is opened for every year from the year of `--period-start` to the year of
`--period-end` inclusive, and every one of them must exist. A period ending on
`2021-01-01` therefore requires the 2021 store. For the last year of an
archive, pass the stores explicitly with `--input`.

`slurm/inference.sbatch` runs the configuration file on one GPU; set
`--account` and the environment name before use.

### Options

| Flag | Default | Notes |
|---|---|---|
| `--period-start` | none | first hour included |
| `--period-end` | none | **exclusive**; to include 31 December, give 1 January |
| `--batch-size` | 2 | affects float32 results in the last bits; keep it fixed when comparing runs |
| `--num-workers` | 2 | use 0 on machines with few cores |
| `--accelerator` | `auto` | `gpu` when CUDA is present |
| `--format` | from extension | `.nc` gives NetCDF, `.zarr` gives Zarr |
| `--time-bounds` | `legacy` | `legacy` is `[t−10min, t]`, as published; `centred` is `[t−5min, t+5min]` |
| `--keep-eps-in-total` | off | reproduces the archived output; see section 11 |
| `--exclude-hours` | none | hours to drop, as `YYYY-MM-DDTHH` |

An existing output at the `--output` path is overwritten without warning.

### How samples are formed

Each prediction needs the hour to be downscaled **and the following hour**.
The final hour of the input therefore has no partner and is skipped: *n* hours
of input give **(n − 1) × 6** output fields. To obtain every hour of a period,
the input must extend at least one hour past its end.

---

## 10. Example usage

A complete session from a clean checkout, using synthetic data. It needs no
external data and no GPU, and exercises preprocessing-independent training,
inference and output writing end to end. A model trained this way has no
predictive skill; the purpose is to confirm that the software works.

```bash
git clone https://github.com/midhuna987/MLdpit.git
cd mldpit
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
mkdir -p demo_data demo_output
```

**1. Create one day of synthetic data.** The hourly field is built as the exact
sum of the 10-minute fields, the same relationship as in the UDAG training
data.

```bash
python - <<'PY'
import numpy as np, pandas as pd, xarray as xr

rng = np.random.default_rng(42)
ny = nx = 64
hours = 24

t10 = pd.date_range("2020-01-01T00:05", periods=hours * 6, freq="10min")
t1h = pd.date_range("2020-01-01T00:30", periods=hours, freq="1h")

p10 = rng.gamma(shape=0.3, scale=0.2, size=(hours * 6, ny, nx)).astype("float32")
p10[p10 < 0.05] = 0.0
p1h = p10.reshape(hours, 6, ny, nx).sum(axis=1)

space = {"rlat": np.linspace(-1.0, 1.0, ny), "rlon": np.linspace(-1.0, 1.0, nx)}
attrs = {"units": "kg m-2", "long_name": "total precipitation"}

def write(values, times, path):
    ds = xr.Dataset(
        {"TOT_PREC": (("time", "rlat", "rlon"), values, attrs)},
        coords={"time": times, **space},
    )
    ds.chunk({"time": 24}).to_zarr(path, mode="w", consolidated=True)

write(p10, t10, "demo_data/train_target_2020_SYNTH_10mnts.zarr")
write(p1h, t1h, "demo_data/train_source_2020_SYNTH_1hr.zarr")
print("synthetic data written")
PY
```

**2. Train for two epochs on CPU.**

```bash
mldpit-train \
  --source-dir demo_data --target-dir demo_data \
  --source-template "train_source_{year}_{experiment}_1hr.zarr" \
  --target-template "train_target_{year}_{experiment}_10mnts.zarr" \
  --experiment SYNTH \
  --train-years 2020 --val-years 2020 \
  --accelerator cpu --epochs 2 --batch-size 2 --num-workers 0 \
  --checkpoint-dir demo_output/checkpoints --log-dir demo_output/logs
```

A notice that the strategy and device count have been overridden for CPU is
expected.

**3. Downscale.**

```bash
mldpit-infer \
  --checkpoint demo_output/checkpoints/best_weight.ckpt \
  --input demo_data/train_source_2020_SYNTH_1hr.zarr \
  --accelerator cpu --num-workers 0 --batch-size 2 \
  --output demo_output/demo_prediction.nc
```

This reports 23 hours in and 138 ten-minute fields out: the 24th hour has no
successor (section 9).

**4. Check the output.**

```bash
python - <<'PY'
import numpy as np, xarray as xr

src  = xr.open_zarr("demo_data/train_source_2020_SYNTH_1hr.zarr")["TOT_PREC"]
pred = xr.open_dataset("demo_output/demo_prediction.nc")["TOT_PREC"]

n = pred.sizes["time"] // 6
block  = pred.values.reshape(n, 6, *pred.shape[1:]).sum(axis=1)
hourly = src.values[:n]

print("fields written:          ", pred.sizes["time"])
print("first six timestamps:    ", [str(t)[11:16] for t in pred.time.values[:6]])
print("max |sum of six - hourly|:", float(np.abs(block - hourly).max()), "kg m-2")
print("minimum predicted value: ", float(pred.values.min()), "kg m-2")
PY
```

Expected: 138 fields; timestamps `00:05` to `00:55`; a maximum difference of
order 10⁻⁶ kg m⁻² (float32 rounding); and a minimum of 0.

---

## 11. Output description

### Structure

| Variable | Dimensions | Description |
|---|---|---|
| `TOT_PREC` | `(time, rlat, rlon)` | 10-minute precipitation, `kg m-2` |
| `time` | `(time)` | interval centres: `:05, :15, … :55` |
| `time_bnds` | `(time, bnds)` | interval bounds (see below) |
| `rlat`, `rlon` | | rotated-pole dimension coordinates |
| `lat`, `lon` | `(rlat, rlon)` | geographical coordinates, when the source provides them |
| `lat_bnds`, `lon_bnds` | `(rlat, rlon, nv)` | cell corners, when the source provides them |
| `rotated_pole` | scalar | grid mapping, when the source provides it |

The output follows the CF conventions: `time` carries `bounds = "time_bnds"`;
`time` and `time_bnds` share the units `minutes since <first bound>`; the
`grid_mapping` attribute of `TOT_PREC` resolves to a variable present in the
file; and no coordinate or bounds variable carries a `_FillValue`. `cdo info`
on the output runs without warnings. Variable and global attributes are
carried over from the hourly input.

### The `time_bnds` convention

Timestamps are interval centres, so a field stamped `00:05` represents
`00:00`–`00:10`. The files produced for the manuscript use bounds of
`[t − 10 min, t]`, which label that field `23:55`–`00:05`.

Both conventions are available; the default reproduces the published files:

```bash
mldpit-infer ... --time-bounds legacy    # [t-10min, t]      default, as published
mldpit-infer ... --time-bounds centred   # [t-5min, t+5min]  consistent with the timestamps
```

Precipitation values are identical under both. Anyone matching the output
against observations by interval should check which convention their files
carry.

### Dry cells and the ε floor

The transform `log10(P + ε)` means that `10^x` reconstructs `P + ε`. Passing
that value to the constraint gives a completely dry hour a mass of ε to
distribute, so its six outputs are small positive values summing to
10⁻⁶ kg m⁻².

The released inference path subtracts ε from the hourly total before the
constraint distributes it, so dry hours produce zeros. The change in any output
cell is bounded by ε, that is, by 10⁻⁶ kg m⁻².


### Mass conservation in practice

Measured with the released code:

| | |
|---|---|
| Maximum absolute error, sum of six fields against the hourly total | 2.4 × 10⁻⁶ kg m⁻² |
| Minimum value | 0 |

The absolute error reproduces to the same digits across machines and grid
sizes. It is the float32 rounding limit of the softmax multiplication, and it
holds for an untrained network as for the trained one, because the constraint
is structural.

### Reading the output

```python
import xarray as xr
ds = xr.open_dataset("predictions_2024-07.nc")
print(ds)

hourly = ds["TOT_PREC"].resample(time="1h").sum()     # recovers the hourly input
daily  = ds["TOT_PREC"].resample(time="1D").sum()
series = ds["TOT_PREC"].sel(rlat=0.0, rlon=0.0, method="nearest")
```

---

## 12. Citation

Please cite both the software and the paper.

**Software**

```
Thayyil Mandodi, M., Arnold, C., Keil, P., Greenberg, D. S., Geyer, B., and
Hagemann, S. (2026). MLdpit: deep learning temporal downscaling of
precipitation (version 1.0.0) [Software].
https://github.com/midhuna987/MLdpit
```

The paper is under review. Once it is published, and once the software and
weights are archived on Zenodo, this section will give all three citations with
their DOIs. When citing a specific version, use that version's DOI; the concept
DOI always resolves to the latest version.

Machine-readable citation metadata is in `CITATION.cff`.

---

## Support

Issues and questions: https://github.com/midhuna987/MLdpit/issues

Corresponding author: Midhuna Thayyil Mandodi (midhuna.thayyil@hereon.de)
