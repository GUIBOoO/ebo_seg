import os
import glob
from sklearn.model_selection import train_test_split

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    RandCropByPosNegLabeld,
    ToTensord,
    MapTransform
)
from monai.data import Dataset, DataLoader,pad_list_data_collate

import nibabel as nib

print("nibabel version:", nib.__version__)

base_dir = os.environ['PYTHON_DATA_DIR']
print(f"Path used by Python : {base_dir}")

class LoadSliced(MapTransform):
    """
    Charge une tranche 2D spécifique d'un volume 3D.
    """
    def __call__(self, data):
        img_path = data["image"]
        lbl_path = data["label"]
        idx = data["slice_index"]

        img = nib.load(img_path).get_fdata()[:, :, idx]
        lbl = nib.load(lbl_path).get_fdata()[:, :, idx]

        img = img[None, ...] 
        lbl = lbl[None, ...]

        return {"image": img.astype("float32"), "label": lbl.astype("float32")}


def get_acdc_slices(data_dir):
    patient_dirs = sorted(glob.glob(os.path.join(data_dir, "patient*")))
    data = []

    for patient in patient_dirs:
        images = sorted(glob.glob(os.path.join(patient, "*frame*.nii.gz")))
        for img_path in images:
            if "_gt" in img_path:
                continue
            label_path = img_path.replace(".nii.gz", "_gt.nii.gz")
            if not os.path.exists(label_path):
                continue

            img_nii = nib.load(img_path)
            num_slices = img_nii.shape[2]
            for i in range(num_slices):
                data.append({
                    "image": img_path,
                    "label": label_path,
                    "slice_index": i
                })

    return data

def get_transforms():
    train_transforms = Compose([
        LoadSliced(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image", "label"])
    ])

    val_transforms = Compose([
        LoadSliced(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image", "label"])
    ])

    return train_transforms, val_transforms

def get_dataloaders(base_dir=base_dir, batch_size=4):
    train_path = os.path.join(base_dir, "training")
    test_path = os.path.join(base_dir, "testing")

    train_data = get_acdc_slices(train_path)
    test_data = get_acdc_slices(test_path)

    print(f"Total train/test slices: {len(train_data)}/{len(test_data)}")

    train_data_split, val_data_split = train_test_split(
        train_data, test_size=0.2, random_state=42
    )

    print(f"Train: {len(train_data_split)}, Val: {len(val_data_split)}")

    train_transforms, val_transforms = get_transforms()

    train_ds = Dataset(train_data_split, transform=train_transforms)
    val_ds = Dataset(val_data_split, transform=val_transforms)
    test_ds = Dataset(test_data, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_list_data_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=pad_list_data_collate)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=pad_list_data_collate)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(base_dir, batch_size=4)

    batch = next(iter(train_loader))
    print("Batch image shape:", batch["image"].shape)
    print("Batch label shape:", batch["label"].shape)