# MLdpit

**Deep learning temporal downscaling of precipitation: hourly to 10-minute.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

MLdpit takes one field of hourly precipitation and produces six fields of 10-minute precipitation for that hour. 
A U-Net then predicts how the total for the hour is distributed over time. A softmax constraint ensures that the 
sum of the six fields exactly matches the original hourly field, meaning that mass conservation is a property of
the architecture rather than a penalty in the loss.

The model was trained using ICON simulations on a rotated-pole grid measuring 501 × 501, and pretrained weights 
will be released alongside it. Given hourly precipitation on that grid — for example, from the CORDEX archive 
— those weights generate 10-minute fields without any further training.

---

## Quick start

```bash
git clone https://github.com/midhuna987/mldpit.git
cd mldpit

python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps


```

---

## Downscaling your own data

With a trained checkpoint — the released weights once available or one you have trained yourself:

```bash
mldpit-infer \
  --checkpoint weights/best_weight.ckpt \
  --input-dir  /path/to/hourly/zarr \
  --input-template "train_source_{year}_1hr.zarr" \
  --period-start 2024-01-01 \
  --period-end   2024-01-02 \
  --output prediction_2024-01-01.nc
```

`--period-end` is **exclusive**: to include 31 December, give 1 January of the
following year.

Input must be a Zarr store containing a variable named `TOT_PREC` (or
`TOT_PREC_10min`) with dimensions `(time, rlat, rlon)`, stamped at interval
centres — `HH:30` for hourly data. Output is NetCDF or Zarr, chosen by the
extension of `--format`, and is CF-conformant: `time` carries a `bounds`
attribute, the rotated-pole grid mapping is carried across, and coordinate
bounds are preserved where the source provides them.


## Pretrained weights

The weights used for the manuscript will be archived on Zenodo under
CC-BY-4.0, with their own DOI, after the first round of peer review. Until
then they are available to editors and reviewers on request from the
corresponding author.

---

## Requirements

Python 3.11, and the packages are in requirements.txt
The published results were produced with torch 2.7.1, PyTorch Lightning 2.5.2,
xarray 2025.7.1, zarr 3.1.0 and numpy 2.1.3, on the DKRZ Levante system with
NVIDIA A100 GPUs.

Inference runs on CPU. A single hour on the 501 × 501 grid takes a few minutes;
a GPU is worth having for long periods, and is effectively required for
training.



## Repository layout

```
mldpit/
├── README.md
├── User_Manual.md           full documentation
├── requirements.txt         exact versions used for the published results
├── environment.yml          conda equivalent
├── LICENSE                  MIT
├── pyproject.toml           package definition
 ── CITATION.cff
├── src/mldpit/              the package
│   ├── unet.py              network definition
│   ├── model.py             LightningModule and the mass-conservation constraint
│   ├── dataset.py           lazy Zarr datasets
│   ├── datamodule.py        train and validation loaders
│   ├── preprocess.py        NetCDF to Zarr conversion
│   ├── train.py             mldpit-train
│   ├── inference.py         mldpit-infer
│   └── config.py            YAML configuration shared by the entry points
├── configs/                 example YAML configurations
├── slurm/                   SLURM job templates                 
```

Three command-line entry points are installed: `mldpit-preprocess`,
`mldpit-train` and `mldpit-infer`. Each accepts `--config file.yaml`, whose
keys set the defaults for the corresponding flags.


## Data availability

**Hourly precipitation** — the predictor — will be publicly available through
the CORDEX repository.

**10-minute precipitation** — the training target — was generated within the
UDAG project.

- Inference needs only the released weights and hourly
  input. The complete training implementation is in this repository and can be
  read, audited and run against any pair of hourly and sub-hourly datasets a
  user does have.




## Citation

Please cite the software and the paper.

```
Thayyil Mandodi, M., Arnold, C., Keil, P., Greenberg, D. S., Geyer, B., and
Hagemann, S. (2026). MLdpit: deep learning temporal downscaling of
precipitation (version 1.0.0) [Software].
https://github.com/midhuna987/MLdpit
```

A DOI will be added here once the software is archived on Zenodo, and the
paper reference once it is published.

---

## License

MIT, for the code in this repository; see LICENSE. The pretrained
weights, when released, will carry CC-BY-4.0.

## Acknowledgements

Training data were generated within the UDAG project. Computations were
performed on the Levante system at the Deutsches Klimarechenzentrum (DKRZ).
