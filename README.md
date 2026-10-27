# ESoLo

This repository contains the main implementation of **ESoLo**, including covariance-based synthetic EEG generation, model training, and evaluation on real cortico-cortical evoked potential (CCEP) recordings.

## 1. Download the CCEP-COREG dataset

Download the CCEP-COREG dataset using the instructions and links provided in the official repository:

- Dataset repository: https://github.com/iTCf/ccepcoreg
- Dataset record: follow the OSF dataset link provided in the repository above (https://osf.io/wsgzp/overview).

Place the downloaded and preprocessed dataset under `data/ccep_coreg/`. The EEG epochs used by this code must be stored as NumPy arrays (`.npy`) using the following structure:

```text
data/
└── ccep_coreg/
    └── derivatives/
        └── epochs/
            ├── sub-16/
            │   └── eeg/
            │       └── *.npy
            ├── sub-21/
            │   └── eeg/
            │       └── *.npy
            └── ...
```

Each EEG array is expected to contain epochs arranged as `(trials, channels, time points)`. The current experiments use 185 EEG channels, a sampling frequency of 1,000 Hz, and the first 1,000 time points of each epoch.

## 2. Prepare the leadfield files

Subject-specific leadfields were computed in Brainstorm using the DUNEuro finite-element method (FEM), with anatomical MRI segmentation performed using CHARM in SimNIBS 4 and registered EEG electrode positions.

Generate subject-specific leadfields following this workflow, export them as `.mat` files with the structure below, and place them under:

```text
data/ccep_coreg_lf/
```

Use the naming convention `{subject}_lf_vol.mat`, for example `sub-16_lf_vol.mat`.

Each `.mat` file must contain a structure named `lf` with the following fields:

- `lf.pos`: source coordinates in millimetres, shaped `(n_sources, 3)`;
- `lf.leadfield`: leadfield matrix, shaped `(n_channels, n_sources, 3)`;
- `lf.orientation`: source orientations;
- `lf.chanlocs`: EEG channel information.

The subjects configured in the scripts are `sub-16`, `sub-21`, `sub-23`, `sub-27`, and `sub-33`.

## 3. Stimulation information

The stimulation metadata used for evaluation is included at:

```text
info/ccep_coreg_stimulation_information.xlsx
```

The required columns are:

```text
subject, file, stim_channel, stim_intensity, stim_x, stim_y, stim_z
```

Atlas-region and atlas-distance columns are optional and are used when available.

## 4. Generate synthetic EEG

Generate the covariance-based synthetic training dataset for one subject:

```bash
python code/gen_cov_based_dataset.py --subject sub-16
```

The generated dataset is saved under:

```text
data/syn_cov/vol/sub-16/
```

Repeat the command for each required subject. A small test run can be generated with:

```bash
python code/gen_cov_based_dataset.py --subject sub-16 --debug
```

## 5. Train ESoLo

Train ESoLo using covariance-based synthetic EEG and the Gaussian classification plus leadfield-informed objective:

```bash
python code/train_esolo.py \
  --subject sub-16 \
  --data_mode syn_cov \
  --loss_mode leadfield
```

The best checkpoint is saved under:

```text
results/dl_train_gaussian_cov/vol/sub-16/ESoLo/syn_cov/leadfield_0.5/best_model.pt
```

For a short test run, add `--debug`.

## 6. Test ESoLo on real CCEP recordings

Evaluate a trained subject-specific model:

```bash
python code/dl_esolo_test.py \
  --subject sub-16 \
  --data_mode syn_cov \
  --loss_mode leadfield
```

To evaluate all configured subject-specific models:

```bash
python code/dl_esolo_test.py \
  --subject all \
  --data_mode syn_cov \
  --loss_mode leadfield
```

Evaluation results are written to:

```text
results/dl_real_evaluation/vol/ESoLo/syn_cov/leadfield/
```

## Expected project structure

```text
eSoLo_model/
├── code/
│   ├── gen_cov_based_dataset.py
│   ├── train_esolo.py
│   └── dl_esolo_test.py
├── data/
│   ├── ccep_coreg/
│   ├── ccep_coreg_lf/
│   ├── ccep_coreg_pial/
│   └── syn_cov/
├── info/
│   └── ccep_coreg_stimulation_information.xlsx
├── utils/
│   └── models/
│       └── ESoLo.py
└── README.md
```

The scripts are stored in `code/`, while `data/`, `info/`, `results/`, and
`utils/` remain in the project root. Run all commands from the
`eSoLo_model/` directory.

The original CCEP-COREG recordings are not redistributed with this code. Please follow the dataset owners' access, citation, and usage requirements.


## License

The ESoLo source code is released under the MIT License. See `LICENSE` for details.

The CCEP-COREG dataset and derived materials remain subject to their original terms of use and are not covered by this license.
