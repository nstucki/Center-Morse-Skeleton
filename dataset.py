import os
from glob import glob
import pandas as pd
import torch
from torch.utils.data import DataLoader
import monai
from monai.data import list_data_collate, decollate_batch
from monai.transforms import (
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    ScaleIntensityd,
    EnsureTyped,
    RandFlipd,
    RandRotated,
    RandRotate90d,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    LambdaD,
)

def create_data_loaders(config, dataconfig):
    # create a temporary directory and 40 random image, mask pairs
    data_path = dataconfig.DATA.DATA_PATH

    if dataconfig.DATA.DATASET == "topcow_ct" or dataconfig.DATA.DATASET == "topcow_mr":
        images = sorted(glob(os.path.join(data_path+'images', "*"+dataconfig.DATA.KEY+"*"+dataconfig.DATA.FORMAT)))
        segs = sorted(glob(os.path.join(data_path+'labels', "*"+dataconfig.DATA.KEY+"*"+dataconfig.DATA.FORMAT)))
        
        # train and validation files
        train_files = [{"img": img, "seg": seg} for img, seg in zip(images[:dataconfig.DATA.TRAIN_SAMPLES], segs[:dataconfig.DATA.TRAIN_SAMPLES])]
        val_files = [{"img": img, "seg": seg} for img, seg in zip(images[-dataconfig.DATA.VAL_SAMPLES:], segs[-dataconfig.DATA.VAL_SAMPLES:])]

        # define transforms for image and segmentation
        train_transforms = Compose(
            [
                LoadImaged(keys=["img", "seg"]),
                Spacingd(keys=["img", "seg"], pixdim=tuple(dataconfig.DATA.PIXDIM), mode=("bilinear", "nearest")),
                EnsureChannelFirstd(keys=["img", "seg"]), # need to check for new dataset
                Orientationd(keys=["img", "seg"], axcodes="RAS"),
                ScaleIntensityd(keys=["img"]), # doing normalisation here :)
                RandCropByPosNegLabeld(
                    keys=["img", "seg"],
                    label_key="seg",
                    spatial_size=dataconfig.DATA.IMG_SIZE,
                    pos=10,
                    neg=1,
                    num_samples=dataconfig.DATA.NUM_PATCH,
                ),
                #RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 1]),
                EnsureTyped(keys=["img", "seg"]),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["img", "seg"]),
                Spacingd(keys=["img", "seg"], pixdim=tuple(dataconfig.DATA.PIXDIM), mode=("bilinear", "nearest")),
                EnsureChannelFirstd(keys=["img", "seg"]),
                Orientationd(keys=["img", "seg"], axcodes="RAS"),
                ScaleIntensityd(keys=["img"]), # doing normalisation here :)
                EnsureTyped(keys=["img", "seg"]),
            ]
        )

    elif dataconfig.DATA.DATASET == "cremi_boundaries" or dataconfig.DATA.DATASET == "SynBlobDataHoles":

        if str(dataconfig.DATA.TRAIN_SAMPLES).isnumeric():
            images = sorted(glob(f"{data_path}/images/*{dataconfig.DATA.FORMAT}"))
            segs = sorted(glob(f"{data_path}/labels/*{dataconfig.DATA.FORMAT}"))
            train_files = [{"img": img, "seg": seg} for img, seg in zip(images[:dataconfig.DATA.TRAIN_SAMPLES], segs[:dataconfig.DATA.TRAIN_SAMPLES])]
            val_files = [{"img": img, "seg": seg} for img, seg in zip(images[-dataconfig.DATA.VAL_SAMPLES:], segs[-dataconfig.DATA.VAL_SAMPLES:])]

        else:
            train_split = pd.read_csv(f"{data_path}/{dataconfig.DATA.TRAIN_SAMPLES}")
            train_files = [{"img": f"{data_path}/{img}", "seg": f"{data_path}/{seg}"} for img, seg in zip(train_split["images"], train_split["labels"])]
            val_split = pd.read_csv(f"{data_path}/{dataconfig.DATA.VAL_SAMPLES}")
            val_files = [{"img": f"{data_path}/{img}", "seg": f"{data_path}/{seg}"} for img, seg in zip(val_split["images"], val_split["labels"])]

        # define transforms for image and segmentation
        train_transforms = Compose(
            [
                LoadImaged(keys=["img", "seg"], image_only=True),
                EnsureChannelFirstd(keys=["img"], channel_dim=0 if dataconfig.DATA.IMAGE_HAS_CHANNEL_DIMENSION else 'no_channel'),
                EnsureChannelFirstd(keys=["seg"], channel_dim=0 if dataconfig.DATA.LABEL_HAS_CHANNEL_DIMENSION else 'no_channel'),
                # ScaleIntensityd(keys=["img"]), # doing normalisation here :)
                RandCropByPosNegLabeld(
                    keys=["img", "seg"],
                    label_key="seg",
                    spatial_size=dataconfig.DATA.IMG_SIZE,
                    pos=10,
                    neg=1,
                    num_samples=dataconfig.DATA.NUM_PATCH,
                    allow_smaller=False,
                ),
                #RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 1]),
                EnsureTyped(keys=["img", "seg"]),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["img", "seg"], image_only=True),
                EnsureChannelFirstd(keys=["img"], channel_dim=0 if dataconfig.DATA.IMAGE_HAS_CHANNEL_DIMENSION else 'no_channel'),
                EnsureChannelFirstd(keys=["seg"], channel_dim=0 if dataconfig.DATA.LABEL_HAS_CHANNEL_DIMENSION else 'no_channel'),
                # ScaleIntensityd(keys=["img"]), # doing normalisation here :)
                EnsureTyped(keys=["img", "seg"]),
            ]
        )

    elif dataconfig.DATA.DATASET == "MRBrainS":
        train_images = sorted(glob(f"{data_path}/train/*/*/T1_resampled{dataconfig.DATA.FORMAT}"))
        train_segs = sorted(glob(f"{data_path}/train/*/segm_bin{dataconfig.DATA.FORMAT}"))

        train_files = [{"img": img, "seg": seg} for img, seg in zip(train_images, train_segs)]

        val_images = sorted(glob(f"{data_path}/test/*/*/T1_resampled{dataconfig.DATA.FORMAT}"))
        val_segs = sorted(glob(f"{data_path}/test/*/segm_bin{dataconfig.DATA.FORMAT}"))

        val_files = [{"img": img, "seg": seg} for img, seg in zip(val_images, val_segs)]

        # define transforms for image and segmentation
        train_transforms = Compose(
            [
                LoadImaged(keys=["img", "seg"]),
                # Spacingd(keys=["img", "seg"], pixdim=tuple(dataconfig.DATA.PIXDIM), mode=("bilinear", "nearest")),
                EnsureChannelFirstd(keys=["img", "seg"]), # need to check for new dataset
                Orientationd(keys=["img", "seg"], axcodes="RAS"),
                RandFlipd(keys=["img", "seg"], prob=0.5, spatial_axis=0),
                NormalizeIntensityd(keys=["img"]), # doing normalisation here :)
                RandCropByPosNegLabeld(
                    keys=["img", "seg"],
                    label_key="seg",
                    spatial_size=dataconfig.DATA.IMG_SIZE,
                    pos=10,
                    neg=1,
                    num_samples=dataconfig.DATA.NUM_PATCH,
                ),
                #RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 1]),
                EnsureTyped(keys=["img", "seg"]),
                RandScaleIntensityd(keys="img", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="img", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["img", "seg"]),
                # Spacingd(keys=["img", "seg"], pixdim=tuple(dataconfig.DATA.PIXDIM), mode=("bilinear", "nearest")),
                EnsureChannelFirstd(keys=["img", "seg"]),
                Orientationd(keys=["img", "seg"], axcodes="RAS"),
                NormalizeIntensityd(keys=["img"]), # doing normalisation here :)
                EnsureTyped(keys=["img", "seg"]),
            ]
        )
    
    elif dataconfig.DATA.DATASET == "FETA":
        images = sorted(glob(f"{data_path}/*/*/*T2w{dataconfig.DATA.FORMAT}"))
        segs = sorted(glob(f"{data_path}/*/*/*dseg_bin{dataconfig.DATA.FORMAT}"))
        
        # train and validation files
        train_files = [{"img": img, "seg": seg} for img, seg in zip(images[:dataconfig.DATA.TRAIN_SAMPLES], segs[:dataconfig.DATA.TRAIN_SAMPLES])]
        val_files = [{"img": img, "seg": seg} for img, seg in zip(images[-dataconfig.DATA.VAL_SAMPLES:], segs[-dataconfig.DATA.VAL_SAMPLES:])]

        # define transforms for image and segmentation
        train_transforms = Compose(
            [
                LoadImaged(keys=["img", "seg"]),
                # Spacingd(keys=["img", "seg"], pixdim=tuple(dataconfig.DATA.PIXDIM), mode=("bilinear", "nearest")),
                EnsureChannelFirstd(keys=["img", "seg"]), # need to check for new dataset
                Orientationd(keys=["img", "seg"], axcodes="RAS"),
                RandFlipd(keys=["img", "seg"], prob=0.5, spatial_axis=0),
                RandRotated(keys=["img", "seg"], prob=0.5, range_x=0.15, range_y=0.15, range_z=0.15, mode=("bilinear", "nearest")),
                NormalizeIntensityd(keys=["img"]), # doing normalisation here :)
                RandCropByPosNegLabeld(
                    keys=["img", "seg"],
                    label_key="seg",
                    spatial_size=dataconfig.DATA.IMG_SIZE,
                    pos=2,
                    neg=1,
                    num_samples=dataconfig.DATA.NUM_PATCH,
                ),
                #RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 1]),
                EnsureTyped(keys=["img", "seg"]),
                RandScaleIntensityd(keys="img", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="img", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["img", "seg"]),
                # Spacingd(keys=["img", "seg"], pixdim=tuple(dataconfig.DATA.PIXDIM), mode=("bilinear", "nearest")),
                EnsureChannelFirstd(keys=["img", "seg"]),
                Orientationd(keys=["img", "seg"], axcodes="RAS"),
                NormalizeIntensityd(keys=["img"]), # doing normalisation here :)
                EnsureTyped(keys=["img", "seg"]),
            ]
        )
    elif dataconfig.DATA.DATASET == "platelet":
        train_image = os.path.join(data_path, dataconfig.DATA.TRAIN_SAMPLES)
        val_image = os.path.join(data_path, dataconfig.DATA.VAL_SAMPLES)
        train_label = train_image.replace("images.npy", "labels.npy")
        val_label = val_image.replace("images.npy", "labels.npy")

        train_files = [{"img": train_image, "seg": train_label}]
        val_files = [{"img": val_image, "seg": val_label}]

        # define transforms for image and segmentation
        train_transforms = Compose(
            [
                LoadImaged(keys=["img", "seg"]),
                EnsureChannelFirstd(keys=["img", "seg"]), # need to check for new dataset
                ScaleIntensityd(keys=["img"]), # doing normalisation here :)
                RandFlipd(keys=["img", "seg"], prob=0.5, spatial_axis=0),
                RandCropByPosNegLabeld(
                    keys=["img", "seg"],
                    label_key="seg",
                    spatial_size=dataconfig.DATA.IMG_SIZE,
                    pos=1,
                    neg=1,
                    num_samples=dataconfig.DATA.NUM_PATCH,
                    allow_smaller=False,
                ),
                #RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 1]),
                # Apply a lambda to transform seg so that only label==1 remains
                LambdaD(keys=["seg"], func=lambda seg: (seg == 1).astype(float)),
                EnsureTyped(keys=["img", "seg"]),
                RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[1, 2]),
                RandScaleIntensityd(keys="img", factors=0.1, prob=0.25),
                RandShiftIntensityd(keys="img", offsets=0.1, prob=0.25),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["img", "seg"]),
                EnsureChannelFirstd(keys=["img", "seg"]),
                ScaleIntensityd(keys=["img"]), # doing normalisation here :)
                # Apply a lambda to transform seg so that only label==1 remains
                LambdaD(keys=["seg"], func=lambda seg: (seg == 1).astype(float)),
                EnsureTyped(keys=["img", "seg"]),
            ]
        )


    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(
        train_ds,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=config.TRAIN.NUM_WORKERS,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds,
                            batch_size=config.TRAIN.VAL_BATCH_SIZE,
                            num_workers=config.TRAIN.NUM_WORKERS,
                            collate_fn=list_data_collate)
    
    return train_loader, val_loader