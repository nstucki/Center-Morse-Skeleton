from collections import defaultdict, namedtuple
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Callable, Dict, Generator, NamedTuple, List
from train import dict2obj
from argparse import ArgumentParser
import yaml
from glob import glob
import pandas
import os
import sys
sys.path.append("./metrics/Betti-matching-3D/build")
import betti_matching
from monai.data import list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Spacingd,
    Orientationd,
    Compose,
    LoadImaged,
    ScaleIntensityd,
    EnsureTyped,
    EnsureType,
)
from torch.utils.data import DataLoader
import monai.data
import monai.networks.nets
import torch.utils.data
import numpy as np
import time
from monai.metrics import DiceMetric
from sklearn.metrics import accuracy_score
from metrics.rand import adapted_rand
from metrics.voi import voi
from metrics.cldice import ClDiceMetric
from metrics.betti_matching import BettiMatchingLoss, FiltrationType
import pathlib
import tqdm

parser = ArgumentParser()
parser.add_argument('--model',
                    default=None,
                    help='path the models')
parser.add_argument('--config',
                    default=None,
                    help='config file (.yaml) containing the hyper-parameters for training.')
parser.add_argument('--dataconfig',
                    default=None,
                    help='data config file (.yaml) containing the dataset specific information.')
parser.add_argument('--cuda_visible_device', nargs='*', type=int, default=[0],
                        help='list of index where skip conn will be made')

# DatasetBucket = namedtuple("DatasetBucket", ["data_path", "val_samples", "data_format", "image_has_channel_dimension", "label_has_channel_dimension"])
# RunMetadata = namedtuple("RunMetadata", ["config", "directory"])
# Hyperparameters = namedtuple("Hyperparameters", ["loss_function", "alpha", "filtration", "relative"])
# RunBucket = namedtuple("RunBucket", ["dataset", "run", "checkpoint", "hyperparameters"])
# MetricBucket = namedtuple("MetricBucket", ["metric", "is_binarized", "is_patched"])

# See https://stackoverflow.com/a/30024601
@contextmanager
def elapsed_timer() -> Generator[Callable[[],float], None, None]:
    start = time.perf_counter()
    elapser = lambda: time.perf_counter() - start
    yield lambda: elapser()
    end = time.perf_counter()
    elapser = lambda: end-start

def main(args):
        # Load the dataconfig files
    with open(args.config) as f:
        print('\n*** Config file')
        print(args.dataconfig)
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2obj(config)
    with open(args.dataconfig) as f:
        print('\n*** Dataconfig file')
        print(args.dataconfig)
        dataconfig = yaml.load(f, Loader=yaml.FullLoader)
    dataconfig = dict2obj(dataconfig)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create a temporary directory and 40 random image, mask pairs
    data_path = dataconfig.DATA.DATA_PATH

    images = sorted(glob(os.path.join(data_path+'images', "*"+dataconfig.DATA.KEY+"*"+dataconfig.DATA.FORMAT)))
    segs = sorted(glob(os.path.join(data_path+'labels', "*"+dataconfig.DATA.KEY+"*"+dataconfig.DATA.FORMAT)))
    
    # train and validation files
    train_files = [{"img": img, "seg": seg} for img, seg in zip(images[:dataconfig.DATA.TRAIN_SAMPLES], segs[:dataconfig.DATA.TRAIN_SAMPLES])]
    val_files = [{"img": img, "seg": seg} for img, seg in zip(images[-dataconfig.DATA.VAL_SAMPLES:], segs[-dataconfig.DATA.VAL_SAMPLES:])]
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            Spacingd(keys=["img"], pixdim=tuple(dataconfig.DATA.PIXDIM), mode=("bilinear")),
            EnsureChannelFirstd(keys=["img", "seg"]),
            Orientationd(keys=["img", "seg"], axcodes="RAS"),
            ScaleIntensityd(keys=["img", "seg"]), # doing normalisation here :)
            EnsureTyped(keys=["img", "seg"]),
        ]
    )

    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds,
                            batch_size=config.TRAIN.VAL_BATCH_SIZE,
                            num_workers=config.TRAIN.NUM_WORKERS,
                            collate_fn=list_data_collate)
    post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    dice_loss = DiceMetric()
    betti_matching_loss_relative = BettiMatchingLoss(relative=True, sigmoid=False, filtration_type=FiltrationType.SUPERLEVEL, cpu_batch_size=64)
    betti_matching_loss_nonrelative = BettiMatchingLoss(relative=False, sigmoid=False, filtration_type=FiltrationType.SUPERLEVEL, cpu_batch_size=64)
    cl_dice_loss = ClDiceMetric()

    checkpoint = torch.load(args.model)
    model = monai.networks.nets.UNet(
        spatial_dims=dataconfig.DATA.DIM,
        in_channels=dataconfig.DATA.IN_CHANNELS,
        out_channels=dataconfig.DATA.OUT_CHANNELS,
        channels=config.MODEL.CHANNELS,
        strides=config.MODEL.STRIDES,
        num_res_units=config.MODEL.NUM_RES_UNITS,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    bm_loss_relative = []
    bm_loss_nonrelative = []
    dice = []
    cl_dice = []

    with torch.no_grad():
        for val_data in (val_data_tqdm := tqdm.tqdm(val_loader, leave=False)):
            val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)

            roi_size = tuple(dataconfig.DATA.IMG_SIZE)
            sw_batch_size = 4
            if dataconfig.DATA.IN_CHANNELS == 1:
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
            elif dataconfig.DATA.IN_CHANNELS == 3:
                val_outputs = sliding_window_inference(torch.squeeze(val_images).permute(0,3,1,2), roi_size, sw_batch_size, model)
            val_outputs = post_trans(val_outputs)
            val_outputs = torch.nn.functional.interpolate(val_outputs, val_labels.shape[2:], mode='nearest')
            bm_loss_relative_ = betti_matching_loss_relative(val_outputs, val_labels)
            bm_loss_nonrelative_ = betti_matching_loss_nonrelative(val_outputs, val_labels)
            dice_ = dice_loss(val_outputs, val_labels)
            cl_dice_ = cl_dice_loss(val_outputs, val_labels)

            bm_loss_relative.append(bm_loss_relative_[0].item())
            bm_loss_nonrelative.append(bm_loss_nonrelative_[0].item())
            dice.append(dice_.item())
            cl_dice.append(cl_dice_[0])

    # After each checkpoint that has been evaluated, save the (partial) dataframe to csv
    # save_dataframe(validation_results, f"evaluation/evaluation_{args.base_tag}.csv")
    print(f"BM Loss Relative: {np.mean(bm_loss_relative)}")
    print(f"BM Loss Nonrelative: {np.mean(bm_loss_nonrelative)}")
    print(f"Dice: {np.mean(dice)}")
    print(f"ClDice: {np.mean(cl_dice)}")
    print("Evaluation finished.")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)