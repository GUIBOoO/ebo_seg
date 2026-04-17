# ebo_seg

Scripts minimalistes pour entrainer un U-Net sur un dataset de segmentation stocke dans le `scratch`.

## Structure attendue du dataset

Le script attend une arborescence de ce type :

```text
DATASET_ROOT/
  train/
    images/
      sample_001.png
      sample_002.png
    masks/
      sample_001.png
      sample_002.png
  val/
    images/
      sample_101.png
    masks/
      sample_101.png
```

Contraintes :

- chaque image doit avoir un masque avec le meme nom de fichier
- segmentation binaire : masques en `0/1` ou `0/255`
- segmentation multi-classe : masques avec labels entiers `0..num_classes-1`

## Lancement rapide

```bash
bash train_unet_scratch.sh
```

## Exemple avec chemins explicites

```bash
DATASET_ROOT=/home/guibo/links/scratch/datasets/mon_dataset \
OUTPUT_DIR=/home/guibo/links/scratch/models/ebo_seg_unet \
NUM_CLASSES=1 \
EPOCHS=50 \
BATCH_SIZE=8 \
LR=1e-3 \
bash train_unet_scratch.sh
```

## Slurm

```bash
sbatch slurm_train_unet.sh
```
