import logging
import os
import yaml
import sys
import shutil
import random
import numpy as np
import json
from argparse import ArgumentParser
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
import sys
from shutil import copyfile
from glob import glob
import pandas as pd

import torch
import wandb
import monai
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureType,
)
from monai.visualize import plot_2d_or_3d_image

from dataset import create_data_loaders

from cldice_loss.cldice import soft_cldice, soft_dice_cldice
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--config',
                    default=None,
                    help='config file (.yaml) containing the hyper-parameters for training.')
parser.add_argument('--dataconfig',
                    default=None,
                    help='data config file (.yaml) containing the dataset specific information.')
parser.add_argument('--pretrained', default=None, help='checkpoint of the pretrained model')
parser.add_argument('--resume', default=None, help='checkpoint of the last epoch of the model')
parser.add_argument('--overwrite', action='store_true', help='overwrite the experiment folder')
parser.add_argument('--cuda_visible_device', nargs='*', type=int, default=[0],
                        help='list of index where skip conn will be made')
parser.add_argument('--logging', action='store_true')

class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)
        
def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)

def main(args):
    # Load the config files
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

    # fixing seed for reproducibility
    random.seed(config.TRAIN.SEED)
    np.random.seed(config.TRAIN.SEED)
    torch.random.manual_seed(config.TRAIN.SEED)
    
    if args.resume and args.pretrained:
        raise Exception('Do not use pretrained and resume at the same time.')
    
    # Loss function choice
    if config.LOSS.USE_LOSS == 'Dice':
        if args.pretrained:
            exp_name = config.LOSS.USE_LOSS
        else:
            exp_name = config.LOSS.USE_LOSS + '_scratch'
        loss_function = monai.losses.DiceLoss(sigmoid=False)
    if config.LOSS.USE_LOSS == 'Dice_ClDice':
        if args.pretrained:
            exp_name = config.LOSS.USE_LOSS+'_'+config.LOSS.SKEL_METHOD+'_alpha_'+str(config.LOSS.ALPHA)
        else:
            exp_name = config.LOSS.USE_LOSS+'_'+config.LOSS.SKEL_METHOD+'_alpha_'+str(config.LOSS.ALPHA)+'_scratch'
        loss_function = soft_dice_cldice(mode=config.LOSS.SKEL_METHOD, alpha=config.LOSS.ALPHA)
    if config.LOSS.USE_LOSS == 'ClDice':
        if args.pretrained:
            exp_name = config.LOSS.USE_LOSS+'_'+config.LOSS.SKEL_METHOD
        else:
            exp_name = config.LOSS.USE_LOSS+'_'+config.LOSS.SKEL_METHOD+'_scratch'
        loss_function = soft_cldice(mode=config.LOSS.SKEL_METHOD)

    # Copy config files and verify if files exist
    exp_path = './runs/'+dataconfig.DATA.DATASET+'/'+exp_name
    if os.path.exists(exp_path) and args.overwrite:
        # remove the folder and create a new one
        print('WARNING: Overwriting the experiment folder!')
        print('Path:', exp_path)
        # ask for confirmation from user via terminal
        response = input('Do you want to continue? (y/n): ')
        if response.lower() != 'y':
            print('Exiting...')
            sys.exit()
        shutil.rmtree(exp_path)
        os.makedirs(exp_path)
    elif os.path.exists(exp_path) and args.resume == None:
        raise Exception('ERROR: Experiment folder exist, please delete or use flag --overwrite')
    else:
        try:
            os.makedirs(exp_path)
            copyfile(args.config, os.path.join(exp_path, "config.yaml"))
        except:
            pass
    
    # initialize wandb
    if args.logging:
        wandb.login()
        wandb.init(project="DMT-skeleton", name=dataconfig.DATA.DATASET+'_'+exp_name, resume="allow")
        wandb.config.update(config)
        wandb.config.update(dataconfig)

    # monai.config.print_config()
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    train_loader, val_loader = create_data_loaders(config, dataconfig)
    
    dice_metric = DiceMetric(include_background=True,
                             reduction="mean",
                             get_not_nans=False)
    
    post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on", device)
    model = monai.networks.nets.UNet(
        spatial_dims=dataconfig.DATA.DIM,
        in_channels=dataconfig.DATA.IN_CHANNELS,
        out_channels=dataconfig.DATA.OUT_CHANNELS,
        channels=config.MODEL.CHANNELS,
        strides=config.MODEL.STRIDES,
        num_res_units=config.MODEL.NUM_RES_UNITS,
    ).to(device)
        
    optimizer = torch.optim.AdamW(model.parameters(), config.TRAIN.LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000000, gamma=0.1)   #always check that the step size is high enough
    
    # Resume training
    last_epoch = 0
    if args.resume:
        dic = torch.load(args.resume)
        model.load_state_dict(dic['model'])
        optimizer.load_state_dict(dic['optimizer'])
        scheduler.load_state_dict(dic['scheduler'])
        last_epoch = int(scheduler.last_epoch/len(train_loader))
        
    # Start from pretrained model
    if args.pretrained:
        dic = torch.load(args.pretrained)
        model.load_state_dict(dic['model'])
        
    # start a typical PyTorch training
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()

    for epoch in tqdm(range(last_epoch, config.TRAIN.MAX_EPOCHS)):
        model.train()
        epoch_loss = 0

        wandb_epoch_dict = {}
        for iter_, batch_data in enumerate(tqdm(train_loader)):
            inputs, labels = batch_data["img"].to(torch.float32).to(device), batch_data["seg"].to(torch.float32).to(device)
            optimizer.zero_grad()
            wandb_batch_dict = {}

            if dataconfig.DATA.IN_CHANNELS == 1:
                outputs = model(inputs)
            elif dataconfig.DATA.IN_CHANNELS == 3:
                outputs = model(torch.squeeze(inputs).permute(0,3,1,2))  # for RGB data

            outputs = torch.sigmoid(outputs)
            if config.LOSS.USE_LOSS == 'Dice':
                loss = loss_function(outputs, labels)
            else:
                loss, dic = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

            if args.logging:
                if (iter_+1) % config.TRAIN.LOG_INTERVAL == 0:
                    wandb_batch_dict.update({'train_loss': loss.item()})
                    if config.LOSS.USE_LOSS == 'Dice':
                        wandb_batch_dict.update({'dice_loss': loss.item()})
                    else:
                        for key, val in dic.items():
                            wandb_batch_dict.update({key: val.item()})
                wandb.log(wandb_batch_dict)

        epoch_loss /= len(train_loader)
        epoch_loss_values.append(epoch_loss)

        if (epoch + 1) % config.TRAIN.VAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None

                #betti_distances = []
                for val_data in tqdm(val_loader):
                    val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)
                    roi_size = tuple(dataconfig.DATA.IMG_SIZE)
                    sw_batch_size = 4
                    if dataconfig.DATA.IN_CHANNELS == 1:
                        val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    elif dataconfig.DATA.IN_CHANNELS == 3:
                        val_outputs = sliding_window_inference(torch.squeeze(val_images).permute(0,3,1,2), roi_size, sw_batch_size, model)  # for RGB data
                    # val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    val_outputs_bin = post_trans(val_outputs)
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs_bin, y=val_labels)

                    #for pair in zip(val_outputs,val_labels):
                    #    betti_distances.append(compute_Betti_distance(pair))
                #betti_distance = torch.mean(torch.stack(betti_distances).float())
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)
                dic = {}
                dic['model'] = model.state_dict()
                dic['optimizer'] = optimizer.state_dict()
                dic['scheduler'] = scheduler.state_dict()
                torch.save(dic, './runs/'+dataconfig.DATA.DATASET+'/'+exp_name+'/last_model_dict.pth')
                # if (epoch+1)%(int(config.TRAIN.MAX_EPOCHS/50)) == 0:
                #     torch.save(dic, './runs/'+dataconfig.DATA.DATASET+'/'+exp_name+'/epoch'+str(epoch+1)+'_model_dict.pth')
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(dic, './runs/'+dataconfig.DATA.DATASET+'/'+exp_name+'/best_model_dict.pth')
                
                #if betti_distance < best_betti_distance:
                #    best_betti_distance = betti_distance
                #    best_betti_distance_epoch = epoch + 1
                #    torch.save(dic, 'best_betti_model_'+name+'_dict.pth')
                    #print("saved new best metric model")
                #print(
                #    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                #        epoch + 1, metric, best_metric, best_metric_epoch
                #    )
                #)
                if args.logging:
                    wandb_epoch_dict.update({'val_mean_dice': metric})
                    # log val_images, val_labels and val_outputs as GIF in wandb
                    wandb_epoch_dict.update({"val_images": [wandb.Video((i.cpu().numpy().transpose(3,0,1,2)*255).astype(np.uint8), caption='Sample'+str(i), fps=10) for i in val_images]})
                    wandb_epoch_dict.update({"val_labels": [wandb.Video((i.cpu().numpy().transpose(3,0,1,2)*255).astype(np.uint8), caption='Sample'+str(i), fps=10) for i in val_labels]})
                    wandb_epoch_dict.update({"val_outputs": [wandb.Video((i.cpu().numpy().transpose(3,0,1,2)*255).astype(np.uint8), caption='Sample'+str(i), fps=10) for i in val_outputs_bin]})
                     
                    #wandb_epoch_dict.update({'val_mean_betti': betti_distance})
                    wandb.log(wandb_epoch_dict)

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    wandb.finish()

if __name__ == "__main__":
    args = parser.parse_args()
    if args.cuda_visible_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.cuda_visible_device))
    main(args)