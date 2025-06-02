#!/usr/bin/env python3
"""

"""

FISHER_WEIGHTS_PATH = "/home/ec2-user/Fisher/fisher_weights.pth"
IVH_WEIGHTS_PATH    = "/home/ec2-user/Fisher/ivh_weights.pth"
IVH_THRESHOLD       = 0.50       

import argparse, os, csv, sys, warnings
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from 3dcnn import FisherNet, fisher_preprocess   
from ivh   import IVHNet,   ivh_preprocess       

class SeriesDirDataset(Dataset):
    def __init__(self, dataset_dir):
        self.series_paths = [
            os.path.join(dataset_dir, d)
            for d in sorted(os.listdir(dataset_dir))
            if os.path.isdir(os.path.join(dataset_dir, d))
        ]

    def __len__(self):
        return len(self.series_paths)

    def __getitem__(self, idx):
        path = self.series_paths[idx]
        return path, os.path.basename(path) 


def adjust_fisher(base_pred: int, ivh_prob: float, thresh: float) -> int:
    """
    """
    ivh_present = ivh_prob >= thresh

    if ivh_present:
        if base_pred in (1, 3):
            return base_pred + 1         
    else:  # IVH absent
        if base_pred in (2, 4):
            return base_pred - 1              #
    return base_pred                          


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Predict modified Fisher score using FisherNet + IVHNet fusion"
    )
    parser.add_argument("--dataset_dir", required=True,
                        help="Folder containing one sub-directory per CT series")
    parser.add_argument("--labels_csv", default=None,
                        help="(optional) CSV with ground-truth labels for accuracy")
    parser.add_argument("--out_csv", required=True,
                        help="Destination CSV for predictions")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="≠1 only if your preprocess returns tensors of equal shape")
    parser.add_argument("--ivh_thresh", type=float, default=IVH_THRESHOLD,
                        help="Decision threshold for IVH presence [0-1]")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) load models --------------------------------------------------
    fisher_net = FisherNet().to(device)
    fisher_net.load_state_dict(torch.load(FISHER_WEIGHTS_PATH, map_location=device))
    fisher_net.eval()

    ivh_net = IVHNet().to(device)
    ivh_net.load_state_dict(torch.load(IVH_WEIGHTS_PATH, map_location=device))
    ivh_net.eval()

    dataset = SeriesDirDataset(args.dataset_dir)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=False, num_workers=0)  # keep num_workers=0 for DICOM I/O

    rows = []
    with torch.no_grad():
        for series_batch in loader:
            paths, ids = series_batch

            fisher_inputs = [fisher_preprocess(p) for p in paths]
            fisher_inputs = torch.stack(fisher_inputs).to(device)  # B×C×D×H×W???

            ivh_inputs = [ivh_preprocess(p)   for p in paths]
            ivh_inputs = torch.stack(ivh_inputs).to(device)

            # Forward pass
            fisher_logits = fisher_net(fisher_inputs)              # 
            fisher_preds  = torch.argmax(fisher_logits, dim=1) + 1 # -> TODO: double check how is dim= 1 here and feed ivh as sequence???

            ivh_logits = ivh_net(ivh_inputs).squeeze(1)            # B   (logit)
            ivh_probs  = torch.sigmoid(ivh_logits)                 # B   (prob)

            for sid, base_pred, ivhp in zip(ids, fisher_preds.cpu(), ivh_probs.cpu()):
                final_pred = adjust_fisher(int(base_pred), float(ivhp), args.ivh_thresh)
                rows.append(
                    dict(series_id=sid,
                         base_pred=int(base_pred),
                         ivh_prob=float(ivhp),
                         final_pred=final_pred)
                )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Saved predictions → {args.out_csv}")

    if args.labels_csv and os.path.exists(args.labels_csv):
        labels = pd.read_csv(args.labels_csv)
        merged = out_df.merge(labels, on="series_id", how="inner")
        if len(merged):
            acc = (merged["final_pred"].round().astype(int) == merged["score"].astype(int)).mean()
            print(f"Accuracy on {len(merged)} labelled scans: {acc*100:.2f}%")
        else:
            warnings.warn("No matching series_id between predictions and labels file.")


if __name__ == "__main__":
    main()
