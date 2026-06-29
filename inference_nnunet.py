import os
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


from utils import energy, doctor, SIRC


DEVICE = torch.device("cuda")


RESULTS = Path(
    "/scratch/guibo/nnUNet_results/Dataset001_ACDC/"
    "nnUNetTrainer__nnUNetPlans__2d"
)

PREPROC = Path(
    "/scratch/guibo/nnUNet_preprocessed/Dataset001_ACDC"
)


def fpr95(scores, labels):

    fpr,tpr,_ = roc_curve(labels, scores)

    idx = np.argmin(abs(tpr-0.95))

    return fpr[idx]



all_scores = {
    "energy": [],
    "msp": [],
    "alpha": [],
    "beta": [],
    "sirc": []
}

all_errors=[]



for fold in range(5):

    print("\n===== FOLD",fold,"=====")


    checkpoint = (
        RESULTS /
        f"fold_{fold}" /
        "checkpoint_best.pth"
    )


    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=DEVICE
    )


    predictor.initialize_from_trained_model_folder(
        str(RESULTS),
        use_folds=(fold,),
        checkpoint_name="checkpoint_best.pth"
    )

    print(predictor.dataset_json.keys())
    val_cases = predictor.dataset_json["validation"]


    for case in val_cases:

        img = np.load(
            PREPROC /
            "nnUNetPlans_2d" /
            case +
            ".npy"
        )


        data = torch.tensor(
            img,
            device=DEVICE
        ).float()


        data=data.unsqueeze(0)


        with torch.no_grad():

            logits = predictor.network(data)


        gt_path = (
            PREPROC /
            "gt_segmentations" /
            (case+".nii.gz")
        )


        gt = nib.load(gt_path).get_fdata()

        gt=torch.tensor(
            gt,
            device=DEVICE
        ).long()



        pred = logits.argmax(1)

        error = (
            pred != gt
        ).flatten().cpu().numpy()


        softmax=torch.softmax(
            logits,
            dim=1
        )


        msp = torch.max(
            softmax,
            dim=1
        ).values


        alpha,beta = doctor(softmax)


        e = energy(logits)


        sirc = SIRC(
            msp,
            1,
            e
        )


        scores={

            "energy":e,
            "msp":-msp,
            "alpha":alpha,
            "beta":beta,
            "sirc":-sirc
        }


        for k,v in scores.items():

            all_scores[k].append(
                v.flatten()
                .detach()
                .cpu()
                .numpy()
            )


        all_errors.append(error)



labels=np.concatenate(all_errors)


print("\n========= RESULTS =========")


for k in all_scores:

    scores=np.concatenate(
        all_scores[k]
    )


    print("\n",k)

    print(
        "AUROC:",
        roc_auc_score(
            labels,
            scores
        )
    )


    print(
        "AUPR:",
        average_precision_score(
            labels,
            scores
        )
    )


    print(
        "FPR95:",
        fpr95(
            scores,
            labels
        )
    )