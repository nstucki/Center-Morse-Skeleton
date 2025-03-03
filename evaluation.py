from contextlib import contextmanager
from typing import Callable, Generator
from train import dict2obj
from argparse import ArgumentParser
import yaml
import sys
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
sys.path.append("./metrics/Betti-matching-3D/build")
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureType,
)
from dataset import create_data_loaders
import monai.data
import monai.networks.nets
import torch.utils.data
import numpy as np
import time
from monai.metrics import compute_dice as dice_metric
from monai.metrics import compute_hausdorff_distance as hd_metric
from sklearn.metrics import accuracy_score
from metrics.rand import adapted_rand
from metrics.voi import voi
from metrics.cldice import ClDiceMetric
from metrics.betti_matching import BettiMatchingLoss, FiltrationType
from metrics.betti_error import bett_error
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
        print(args.config)
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2obj(config)
    with open(args.dataconfig) as f:
        print('\n*** Dataconfig file')
        print(args.dataconfig)
        dataconfig = yaml.load(f, Loader=yaml.FullLoader)
    dataconfig = dict2obj(dataconfig)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the data
    _, val_loader = create_data_loaders(config, dataconfig)

    post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

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
    b_error = []
    hd_error = []

    with torch.no_grad():
        for val_data in (val_data_tqdm := tqdm.tqdm(val_loader, leave=False)):
            val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)

            roi_size = tuple(dataconfig.DATA.IMG_SIZE)
            sw_batch_size = 4
            if dataconfig.DATA.IN_CHANNELS == 1:
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
            elif dataconfig.DATA.IN_CHANNELS == 3:
                val_outputs = sliding_window_inference(torch.squeeze(val_images).permute(0,3,1,2), roi_size, sw_batch_size, model)
            val_outputs = torch.nn.functional.interpolate(val_outputs, val_labels.shape[2:], mode='trilinear', align_corners=True)
            val_outputs = post_trans(val_outputs)
            # print(val_outputs.shape, val_labels.shape)
            # bm_loss_relative_ = betti_matching_loss_relative(val_outputs, val_labels)
            bm_loss_nonrelative_ = betti_matching_loss_nonrelative(val_outputs, val_labels)
            dice_ = dice_metric(val_outputs, val_labels)
            cl_dice_ = cl_dice_loss(val_outputs, val_labels)
            bett_error_ = bett_error(val_outputs.cpu().squeeze().numpy(), val_labels.cpu().squeeze().numpy())
            hd_error_ = hd_metric(val_outputs, val_labels)

            # bm_loss_relative.append(bm_loss_relative_[0].item())
            bm_loss_nonrelative.append(bm_loss_nonrelative_[0].item())
            dice.append(dice_.cpu().numpy())
            cl_dice.append(cl_dice_[0])
            b_error.append(bett_error_)
            hd_error.append(hd_error_.cpu().numpy())

    # After each checkpoint that has been evaluated, save the (partial) dataframe to csv
    # save_dataframe(validation_results, f"evaluation/evaluation_{args.base_tag}.csv")
    # print(f"BM Loss Relative: {np.mean(bm_loss_relative)}")
    print(f"Dice: {np.mean(dice)}")
    print(f"ClDice: {np.mean(cl_dice)}")
    print(f"Hausdorff Error: {np.mean(hd_error)}")
    print(f"Betti Error: {np.mean(b_error, axis=0)}")
    print(f"BM Loss Nonrelative: {np.mean(bm_loss_nonrelative)}")
    print("Evaluation finished.")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.cuda_visible_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.cuda_visible_device))
    main(args)