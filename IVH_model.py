import argparse, os, sys, glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pydicom import dcmread
import pylibjpeg

from tqdm import tqdm
from tqdm import trange



import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils.data import Dataset
from albumentations import pytorch
from torch.utils.data import DataLoader

import torchvision
import pretrainedmodels
import cv2


# IVH = intraventricular btw
fields =  ["any", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "epidural"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cutoffs = {
    "any" : 0.21173532,
    "intraparenchymal" : 0.17751145,
    "intraventricular" : 0.15373419,
    "subarachnoid" : 0.20954421,
    "subdural" : 0.20799863,
    "epidural" : 0.059611224000000004
}

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        # TODO: check out the se minimax optim sometime
        # bottleneck layers are 1x1 conv, for dim reduction, 
        # then 3x3 conv for feature transform, then upscale with 1x1
        # TODO: ref: http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006


        backbone = pretrainedmodels.__dict__["resnet50"](num_classes=1000, pretrained="imagenet")

        od = OrderedDict()
        od['conv1'] = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        od['bn1'] = nn.BatchNorm2d(64)
        od['relu'] = nn.ReLU(inplace=True)
        od['maxpool'] = nn.MaxPool2d(kernel_size=3,  stride=2, padding = 1) 
 
        self.layer0 = nn.Sequential(od)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def resnet_forward(self, x):
      x = self.layer0(x)
      x = self.layer1(x)
      x = self.layer2(x)
      x = self.layer3(x)
      x = self.layer4(x)
      return x

class SequenceModel(nn.Module):

    def __init__(self):
        super(SequenceModel, self).__init__()

        self.recurrent = nn.LSTM(input_size=2048, hidden_size=512,
            dropout=0.3, num_layers=2,
            bidirectional=True, batch_first=True)
      
        self.fc = nn.Linear(1024, 6)

    def forward(self, x, seq_len):
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = x.reshape(-1, seq_len, x.size(-1))
        x, _ = self.recurrent(x)
        x = self.fc(x)
        x = x.reshape(-1, x.size(-1))
        return x

class IVHmodel(Resnet):
    def __init__(self):
        super(IVHmodel, self).__init__()
        self.decoder = SequenceModel()

    def forward(self, x, seq_len):
      x = self.resnet_forward(x)
      x = self.decoder(x, seq_len)
      return x

# --------------------------------------------------------------------------- #
# 1. Small helpers                                                            #
# --------------------------------------------------------------------------- #
def listdir_nohidden(path):
  '''Same as os.listdir() except doesn't include hidden files. 

  Parameters
  ----------
  path : str
      Path of directory. 

  Returns
  -------
  list
      A list of all non-hidden files in the diretory, 'path'.
  '''
  ret = []
  for f in os.listdir(path):
      if not f.startswith('.'):
          ret.append(f)
  if len(ret) > 0:
      return ret
  else:
      return


def preprocess(r, window_index):
  '''Preprocesses a single DICOM object into one of the three
    window views.

    Parameters
    ----------
    r : pydicom.FileDataset
        The DICOM file read by dcmread
    window_index : int
        The index for what type of window. 0 for Brain, 1 for Subdural, 
        and 2 for Bony windowing.

    Returns
    -------
    np.array
        The pixel_array of the DICOM file, under the 
        corresponding window. 
    '''

  wl = [40, 75, 600]
  ww = [80, 215, 2800]  
  window_min = wl[window_index]-(ww[window_index] // 2)
  window_max = wl[window_index]+(ww[window_index] // 2)
  # zaxis = -1*r.ImagePositionPatient[2]
  img = (r.pixel_array * r.RescaleSlope) + r.RescaleIntercept
  img = np.clip(img, window_min, window_max)
  img = 255 * ((img - window_min)/ww[window_index])
  img = img.astype(np.uint8)
  return img

def dicom_scan_to_np(path):
    '''Converts CT Scan into stacked RGB-like numpy array.

    Parameters
    ----------
    path : str
        Path to the folder holding the CT Scan (~ 30 DICOM files).

    Returns
    -------
    list
        A list of the DICOM files as RGB-like numpy arrays. 
    '''

    dicom_filenames = [os.path.join(path,file) for file in listdir_nohidden(path) if file.endswith(".dcm")]

    dicom_data = []
    for file in dicom_filenames:
        dicom_data.append(dcmread(file))

    #arrange the CT scan in order
    dicom_data = sorted(dicom_data, key=lambda x: -1*x.ImagePositionPatient[2])

    total_scan = []
    pbar = trange(len(dicom_data)-1, desc="[Preprocessing scan...]", unit="img")
    for dcm, i in zip(dicom_data, pbar):
        imgs = []
        for i in range(3):
            imgs.append(preprocess(dcm,i))
        rgb = np.dstack((imgs[0], imgs[1], imgs[2]))    
        total_scan.append(rgb)
    return total_scan, dicom_data

def load_img_hi(img):
    MEAN = 255 * np.array([0.485, 0.456, 0.406])
    STD = 255 * np.array([0.229, 0.224, 0.225])
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = np.array(img).transpose(-1, 0, 1)
    x = (x - MEAN[:, None, None]) / STD[:, None, None]
    
    return torch.Tensor(x)

def rgb_convert(total_scan):
    '''Takes the ~30 DICOM images from dicom_scan_to_np() and stacks
      them into a (30, *, *, *) tensor for input to the model. Also 
      does the cv2.cvtColor thingy for BGR2RGB.

        Parameters
        ----------
        total_scan : arr
            The array of all processed DICOM np arrays.

        Returns
        -------
        imgs : torch.Tensor
            Converted BGR2RGB stack of tensors         
    ''' 
    imgs = [load_img_hi(img) for img in total_scan]
    imgs = torch.stack(imgs)
    return imgs




# --------------------------------------------------------------------------- #
# 3. Orchestration                                                            #
# --------------------------------------------------------------------------- #
def run_inference(data_dir: Path,
                  df_labels: pd.DataFrame,
                  model,
                  threshold: float = 0.5) -> pd.Series:
    """
    Iterate over every series in df_labels['series_id'] and
    return a pandas Series of 0/1 flags for IVH.
    """
    preds = []
    raw = []
    model.eval()
    with torch.no_grad():
        for sid in df_labels["series_id"]:
            series_path = data_dir / sid
            if not series_path.is_dir():
                sys.stderr.write(f"-  Missing folder {series_path}; marking 0.\n")
                preds.append(0)
                continue

            # --- load & preprocess ------------------------------------------------
            ct, raw_data = dicom_scan_to_np(series_path)
            # print(f'REACHED to rgb convert')

            image = rgb_convert(ct)

            seq_len, c, h, w = image.size()
            image = image.to(device)
            print(f'REACHED TO MODEL {seq_len}')
            output = model(image, seq_len)
            total_outputs = torch.sigmoid(output)
            prediction, _ = torch.max(total_outputs, dim=0, keepdim=True)
            # print(prediction[0])
            # print(f'IVH present?: {prediction[0][2]}')
            # print(f'threhsold : {threshold}')

            preds.append(int(prediction[0][2] >= threshold))
            raw.append(float(prediction[0][2]))



    return pd.Series(preds, name="ivh"), pd.Series(raw, name="raw p")


# --------------------------------------------------------------------------- #
# 4. Main                                                                     #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Add IVH column to labels.csv using IVHmodel")
    parser.add_argument("--data-dir", required=False, type=Path, default=Path("/home/ec2-user/Fisher/FlattenedDataset"),
                        help="Root folder containing one sub-folder per series")
    parser.add_argument("--labels",   required=False, type=Path, default=Path("/home/ec2-user/Fisher/labels.csv"),
                        help="CSV with at least a 'series_id' column")
    parser.add_argument("--model",    required=False, type=Path, default=Path("/home/ec2-user/Fisher/pretrain.pth"),
                        help="Checkpoint or path understood by IVHmodel")
    parser.add_argument("--out",      required=False, type=Path, default=Path("/home/ec2-user/Fisher/labelswIVH.csv"),
                        help="Destination CSV (will overwrite if exists)")
    parser.add_argument("--threshold", type=float, default=0.15373419,
                        help="P(IVH) ≥ threshold ⇒ label 1 (default 0.5)")
    args = parser.parse_args()

    # Safety checks
    if not args.data_dir.is_dir():
        sys.exit(f"Dataset dir {args.data_dir} not found!")
    if not args.labels.is_file():
        sys.exit(f"Labels CSV {args.labels} not found!")
    if not args.model.is_file():
        sys.exit(f"Model file {args.model} not found!")

    # --------------------------------------------------------------------- #
    print("- Loading labels...")
    df = pd.read_csv(args.labels)

    print("- Loading model...")
    model = IVHmodel()
    ckpt = torch.load(args.model, map_location="cuda")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(ckpt.pop('state_dict'))   
    model.to(device)

    print("- Running inference...")
    df["ivh"], df['raw'] = run_inference(args.data_dir, df, model,
                              threshold=args.threshold)


    print("- Saving results ->", args.out)
    df.to_csv(args.out, index=False)
    print("- Done")

if __name__ == "__main__":
    main()
