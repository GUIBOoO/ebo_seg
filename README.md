# ebo_seg

Minimal scripts to train a U-Net on a segmentation dataset stored in scratch.

## Dataset structure

DATASET_ROOT/
  train/
    images/
    masks/
  val/
    images/
    masks/

Each image must have a mask with the same filename.

Binary segmentation: masks are 0/1 or 0/255  
Multi-class segmentation: masks are integer labels (0..num_classes-1)

## Run training

bash train_unet_scratch.sh

## Example

DATASET_ROOT=/path/to/data \
OUTPUT_DIR=/path/to/output \
NUM_CLASSES=1 \
EPOCHS=50 \
BATCH_SIZE=8 \
LR=1e-3 \
bash train_unet_scratch.sh

## Slurm

sbatch slurm_train_unet.sh

## 2D slice datasets (BraTS, REFUGE, CHAOS, CaDIS)

Everything except ACDC is stored as prepared 2D slices, one directory per case:

    <root>/<case_id>/slice_<idx>_img.npy   (C, H, W) uint8 (RGB) or float32
    <root>/<case_id>/slice_<idx>_seg.npy   (H, W)    uint8, labels 0..K-1
    <root>/dataset_info.json               in_channels / num_classes / labels
    <root>/splits.json                     train/val/test case ids

`dataset_info.json` is what train_unet.py reads to default `--in-channels` and
`--num-classes`; `splits.json`, when present, overrides the random case split
(official protocol for REFUGE and CaDIS, patient-level split for CHAOS).

Build them from the released archives (unzip first, see the header of
extract_data_slices.sh):

    DATASET=refuge    bash extract_data_slices.sh
    DATASET=cataracts bash extract_data_slices.sh    # CADIS_EXPERIMENT=1|2|3
    DATASET=chaos     bash extract_data_slices.sh    # writes _ct and _mr

| DATASET     | root                             | channels | classes                | split |
|-------------|----------------------------------|----------|------------------------|-------|
| `brats3`    | `datasets/Brats/data_slices_3ch` | 3        | 4                      | random, per case |
| `refuge`    | `datasets/refuge/data_slices`    | 3        | 3 (bg / disc / cup)    | official folders |
| `cataracts` | `datasets/cataracts/data_slices` | 3        | 8 (Experiment I)       | official splits.txt, per video |
| `chaos_ct`  | `datasets/chaos/data_slices_ct`  | 1        | 2 (bg / liver)         | random, per patient |
| `chaos_mr`  | `datasets/chaos/data_slices_mr`  | 1        | 5 (+ kidneys, spleen)  | random, per patient |

Then train / grid search / test as usual, e.g.

    DATASET=refuge   MODEL=unet LOSS=bound_ebo_ce bash slurm_train_unet.sh
    DATASET=chaos_mr MODEL=unet LOSS=bound_ebo_ce bash grid_search.sh

CaDIS Experiments II and III map their rarest classes to an ignore label, which
none of the losses here support; extract_data_cataracts.py folds those pixels
into one extra trailing class instead. Experiment I (the default) has no ignore
label and reproduces the published protocol as is.

## The same datasets for nnU-Net

Each U-Net dataset has an nnU-Net twin that imports the *same* label remapping,
so the two pipelines are trained on identical targets:

| `DATASET=` | id | nnU-Net dataset | converter | format |
|---|---|---|---|---|
| `acdc` | 1 | `Dataset001_ACDC` | extract_data_acdc.py | NIfTI 3D |
| `brats` | 2 | `Dataset002_BraTS2D` | extract_data_brats.py | NIfTI |
| `refuge` | 4 | `Dataset004_REFUGE` | extract_data_refuge_nnunet.py | RGB png (NaturalImage2DIO) |
| `cataracts` | 5 | `Dataset005_CaDIS` | extract_data_cataracts_nnunet.py | RGB png (NaturalImage2DIO) |
| `chaos_mr` | 6 | `Dataset006_CHAOSMR` | extract_data_chaos_mr_nnunet.py | NIfTI 3D, one file per series |

    python extract_data_refuge_nnunet.py
    python extract_data_cataracts_nnunet.py --experiment 1
    python extract_data_chaos_mr_nnunet.py
    nnUNetv2_plan_and_preprocess -d 4 -c 2d --verify_dataset_integrity   # idem 5, 6

`DATASET=` then drives the id and the default result directories everywhere:

    DATASET=refuge LOSS=bound_ebo bash nnunet_grid_search.sh          # SELECT_BEST=0 to train
    DATASET=refuge SELECT_BEST=1  bash nnunet_grid_search.sh          # 3 selection JSONs
    DATASET=refuge bash inference_nnunet.sh                           # SELECTION=all
    DATASET=refuge MODE=best CRITERION=fpr95 SPLIT=val bash compute_d_matrix_nnunet.sh

Two conversion details worth knowing:

- CHAOS MR keeps its 3D geometry (one volume per series, real voxel spacing);
  `CONFIGURATION=2d` slices it again, exactly like ACDC. Only Train_Sets is
  usable, the released Test_Sets has no ground truth.
- CaDIS has no background class (class 0 is Pupil) but nnU-Net requires a
  `background` label at 0, so every class is shifted by +1 and label 0 is left
  empty. nnU-Net's mean foreground Dice then covers exactly the 8 CaDIS classes,
  i.e. the same set the U-Net pipeline averages over — only the indices differ.