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