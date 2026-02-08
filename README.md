# Deep Learning–Based Quantitative Evaluation of Postoperative Atelectasis

## Overview
This repository contains pretrained nnU-Net v2 model checkpoints and associated resources for automated deep learning–based segmentation supporting quantitative analysis of postoperative atelectasis following right upper lobectomy using paired pre- and postoperative CT scans.

We developed and validated multiple 3D nnU-Net v2 segmentation models to delineate thoracic anatomy across different clinical contexts:

- **Preoperative Lobar Segmentation Model** — segments pulmonary lobes on preoperative CT scans to enable baseline volumetric characterization.
- **Postoperative Lobar Segmentation Model** — optimized for altered postoperative anatomy to support accurate volumetry and atelectasis grading.
- **Airway Segmentation Model** — extracts the bronchial tree, facilitating airway-aware structural analysis and enabling future integration with functional assessments.

Each model release includes:
- **nnU-Net plans** — architecture configurations and preprocessing parameters (required for inference to reconstruct the network architecture, patch size, spacing, and normalization)
- **Dataset configuration files** — dataset fingerprints, modality definitions, and label mappings (required for inference to ensure identical preprocessing)
- **Model checkpoints** — pretrained weights for immediate inference and validation

## Key Features
- Three **nnU-Net v2** segmentation models: preoperative lobar, postoperative lobar, and airway segmentation.
- Automated volumetric quantification of lobar volume loss as a surrogate for atelectasis severity.
- Validation against physician-assigned grades using a standardized 5-point radiological scale.
- Generalizable performance across vendors, scanners, and acquisition protocols.
- External validation on the public **LOLA11 lung lobe segmentation challenge** dataset.

## Dataset
- **Internal Stanford cohort**: 236 patients (2008–2023) with paired CT scans.
- **External dataset**: OttawaChestCT for preoperative training, LOLA11 for external validation.
- DICOMs converted to NIfTI format; preprocessing and postprocessing handled via nnU-Net pipeline and customized postprocessing.

## Model Performance
- **Preoperative model**: Mean Dice ≈ 0.98 (internal), 0.92 (LOLA11).
- **Postoperative model**: Mean Dice ≈ 0.99 (internal).
- Progressive decline in right middle lobe volume correlates with increasing atelectasis severity.

## Figures

<p align="center">
  <img src="images/Figure4.png" alt="Volume Changes vs. Atelectasis Grade" width="600">
</p>

<p align="center">
  <img src="images/Figure7.png" alt="3D Modelling of Lobe Segmentations with progressive grades of atelectasis (top-bottom) and preop-postop (left-right)" width="600">
</p>

## Model Checkpoints

Pretrained model checkpoints, nnU-Net plans, and dataset configuration files for all three models (preoperative, postoperative, and airway) are available for download:

[10.6084/m9.figshare.29877509](https://figshare.com/articles/software/_b_Deep-learning_based_quantitative_evaluation_of_postoperative_atelectasis_following_right_upper_lobectomy_b_/29877509)

## How to Use the Models

### Prerequisites

1. Install [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) following the official instructions.
2. Download the model checkpoints, nnU-Net plans, and dataset configuration files from [Figshare](https://figshare.com/articles/software/_b_Deep-learning_based_quantitative_evaluation_of_postoperative_atelectasis_following_right_upper_lobectomy_b_/29877509).
3. Prepare your input CT scans in NIfTI format (`*.nii.gz`) with the nnU-Net naming convention: `{case_id}_0000.nii.gz`.

### Setting Up the nnU-Net Environment

Set the required environment variables to point to the directories where you placed the downloaded files:

```bash
export nnUNet_raw="/path/to/nnUnet/raw"
export nnUNet_preprocessed="/path/to/nnUnet/preprocessed"
export nnUNet_results="/path/to/nnUnet/results"
```

Place the downloaded plans, dataset configuration files, and checkpoints into the appropriate subdirectories under `nnUNet_results` following the standard nnU-Net v2 folder structure (refer to the [nnU-Net v2 repository](https://github.com/MIC-DKFZ/nnUNet) for full details):

```
nnUNet_results/
└── DatasetXXX_DATASETNAME/
    └── nnUNetTrainer__nnUNetPlans__3d_fullres/
        ├── fold_0/
        │   └── checkpoint_best.pth
        ├── dataset.json
        └── nnUNetPlans.json
```

> **Important:** Files on Figshare are prefixed with a model tag (e.g., `preop_checkpoint_best.pth`, `postop_dataset.json`, `airway_nnUNetPlans.json`). After downloading, **remove the prepended prefix** (`preop_`, `postop_`, or `airway_`) so that filenames match what nnU-Net expects (e.g., `checkpoint_best.pth`, `dataset.json`, `nnUNetPlans.json`).

### Lobe Segmentation Models (Preoperative & Postoperative)

**Preoperative inference** (5-lobe segmentation: RUL, RML, RLL, LUL, LLL):

```bash
nnUNetv2_predict \
  -i /path/to/preop/input_nifti \
  -o /path/to/preop/predictions \
  -d DATASET_ID \
  -c 3d_fullres \
  -f 0 \
  --disable_tta \
  -chk checkpoint_best.pth
```

**Postoperative inference** (remaining lobes after right upper lobectomy: RML, RLL, LL):

```bash
nnUNetv2_predict \
  -i /path/to/postop/input_nifti \
  -o /path/to/postop/predictions \
  -d DATASET_ID \
  -c 3d_fullres \
  -f 0 \
  --disable_tta \
  -chk checkpoint_best.pth
```

> **Note:** Replace `DATASET_ID` with the dataset number corresponding to the downloaded model (refer to the nnU-Net plans file). Use the appropriate `nnUNet_results` path for each model.

**Label mapping (Preoperative):**
| Label | Structure |
|-------|-----------|
| 0 | Background |
| 1 | Right Upper Lobe (RUL) |
| 2 | Right Middle Lobe (RML) |
| 3 | Right Lower Lobe (RLL) |
| 4 | Left Upper Lobe (LUL) |
| 5 | Left Lower Lobe (LLL) |

**Label mapping (Postoperative):**
| Label | Structure |
|-------|-----------|
| 0 | Background |
| 1 | Right Middle Lobe (RML) |
| 2 | Right Lower Lobe (RLL) |
| 3 | Left Lung (LL) |

### Airway Segmentation Model

**Airway inference** (bronchial tree extraction):

```bash
nnUNetv2_predict \
  -i /path/to/airway/input_nifti \
  -o /path/to/airway/predictions \
  -d DATASET_ID \
  -c 3d_fullres \
  -f 0 \
  --disable_tta \
  -chk checkpoint_best.pth
```

> **Note:** Input CT scans should follow the same NIfTI format and naming convention (`{case_id}_0000.nii.gz`).

### Lung Lobe Postprocessing

After inference, we recommend applying the postprocessing steps provided in:
- `LobeInferenceClinicalDataset/Preop/Stan_preop_Dataset_Postprocessing.ipynb` — for preoperative predictions
- `LobeInferenceClinicalDataset/Postop/Stan_postop_Dataset_Postprocessing.ipynb` — for postoperative predictions

These notebooks correct anatomical label assignments (left/right side verification) and remove isolated segmentation artifacts using connected component analysis.

### Airway Metrics Extraction

The `AirwayMetricsExtraction/` folder contains scripts for skeleton-based airway tree analysis and volumetric segmentation from airway model predictions:

- **`analyze_airway_skeleton.py`** — Core skeleton analysis engine. Takes an airway segmentation NIfTI file, skeletonizes it, and traces the airway tree from trachea → carina → right main bronchus → bronchus intermedius → RML. Computes measurements (angles, cross-sectional areas, lengths, volumes) for each segment and can generate 3D visualizations.

- **`batch_airway_skeleton_analysis.py`** — Batch runner that processes an entire folder of airway segmentation NIfTI files in parallel. For each case, it calls `analyze_airway_skeleton.py`, saves a 3D screenshot, extracts RML-specific metrics, and aggregates results into an Excel summary.

- **`segment_airway_volumes.py`** — Converts the skeleton classification back into a full volumetric segmentation. Uses distance transforms to assign every airway voxel to its nearest anatomical region (trachea, right/left main bronchus, RUL, intermedius, RML, RLL, LUL, LLL), producing a labeled NIfTI volume.

<p align="center">
  <img src="images/Figure4a.png" alt="Airway segmentation, skeletonization, and region-labeled 3D reconstructions" width="600">
</p>

<p align="center">
  <img src="images/Figure5.png" alt="Skeleton-based airway metrics: branching angles, cross-sectional areas, and RML measurements" width="600">
</p>

**Before running**, update the following path variables in each script:

| Script | Variable | Description |
|--------|----------|-------------|
| `analyze_airway_skeleton.py` | `nii_file_path` | Path to a single airway segmentation NIfTI file |
| `batch_airway_skeleton_analysis.py` | `input_folder` | Directory containing airway segmentation NIfTI files (`*.nii.gz`) |
| `batch_airway_skeleton_analysis.py` | `output_folder` | Directory for screenshots and Excel summary output |
| `segment_airway_volumes.py` | `nii_file_path` | Path to a single airway segmentation NIfTI file |
| `segment_airway_volumes.py` | `output_dir` | Directory to save the labeled airway NIfTI volume |

```bash
# Single-case skeleton analysis with interactive 3D visualization
python AirwayMetricsExtraction/analyze_airway_skeleton.py

# Batch skeleton analysis of all airway predictions
python AirwayMetricsExtraction/batch_airway_skeleton_analysis.py

# Volumetric segmentation of airway regions
python AirwayMetricsExtraction/segment_airway_volumes.py
```

## Cite:
If you use this code or data, please cite our paper:

**[Full citation to be added upon publication]**
