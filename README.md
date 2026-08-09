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