from itertools import count
import random, math
import numpy as np
import os
from collections import namedtuple
from collections import deque
import os, sys

import time
import argparse

import numpy as np
from scipy.ndimage.interpolation import affine_transform
import nibabel as nib
import random
from skimage.transform import warp
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.utils as vutils
import shutil

# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
# from polyaxon_helper import (
#     get_cluster_def,
#     get_declarations,
#     get_experiment_info,
#     get_task_info,
#     get_tf_config,
#     get_job_info,
#     get_outputs_path,
#     get_outputs_refs_paths,
#     get_data_paths,
#     get_log_level
# )


from pyramid_lstm import PyramidLSTM
from data_generator import RigidDataLoader
from data_generator import DataObject

import multiprocessing
multiprocessing.set_start_method('spawn', True)

"""
initialize weights in network
"""
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()


def eval(args, epoch, model, data_loader, writer):
    model.eval()    
    losses = []
    diff_translation = []
    diff_rotation = []
    ssd_ratio = []
    for batch_idx, (batch_data) in enumerate(data_loader):
        img1, img2_deformed = batch_data.img1, batch_data.img2
        target_transform, img2_orig = batch_data.target_transform, batch_data.img2_orig
        if args.cuda:
            img1, img2_deformed, target_transform = img1.float().cuda(), img2_deformed.float().cuda(), target_transform.float().cuda()
        img1, img2_deformed, target_transform =  Variable(img1),  Variable(img2_deformed),  Variable(target_transform)
        output = model(img1, img2_deformed) 
        loss = F.mse_loss(output, target_transform)
        losses.append(loss.detach().cpu().numpy())
        output = output.detach().cpu().numpy().squeeze()
        target_transform = target_transform.detach().cpu().numpy().squeeze()
        print('target value is:\n', target_transform)
        print('output is:\n', output)
        for i in range(target_transform.shape[0]):
            tr_gt, rot_gt = data_loader.dataset.outputToRotTranData(target_transform[i,:])
            tr_out, rot_out = data_loader.dataset.outputToRotTranData(output[i,:])
            diff_translation.append(np.abs(tr_gt-tr_out))
            # target_rotation = data_loader.dataset.so3_to_euler_angles(target_transform[i,3:6])
            # output_rotation = data_loader.dataset.so3_to_euler_angles(output[i,3:6])
            diff_rotation.append(np.abs(rot_gt-rot_out))
            affine = data_loader.dataset.outputToAffineMatrix(output[i,:])
            A = img2_orig[i,0,...].squeeze().numpy()
            B = data_loader.dataset._transform_image(A, affine)
            C = img2_deformed[i,0,...].detach().cpu().numpy().squeeze()
            ssd_ratio.append((np.square(C - B)).mean()/(np.square(C - A)).mean())

    diff_rotation = np.mean(diff_rotation, axis=0)
    diff_translation = np.mean(diff_translation, axis=0)
    ## save the loss using tensorboardX
    writer.add_scalars('data/eval', { 'loss': np.mean(losses),
                            'x': diff_translation[0],
                            'y': diff_translation[1],
                            'z': diff_translation[2],
                            'x_angle': diff_rotation[0],
                            'y_angle': diff_rotation[1],
                            'z_angle': diff_rotation[2],
                            'ssd_ratio' : np.mean(ssd_ratio)
                            }, epoch)

def train(args, epoch, model, data_loader, optimizer, writer):
    model.train()
    losses = []
    for batch_idx, (batch_data) in enumerate(data_loader):
        img1, img2_deformed = batch_data.img1, batch_data.img2
        target_transform, img2_orig = batch_data.target_transform, batch_data.img2_orig

        if args.cuda:
            img1, img2_deformed, target_transform = img1.float().cuda(), img2_deformed.float().cuda(), target_transform.float().cuda()
        img1, img2_deformed, target_transform =  Variable(img1),  Variable(img2_deformed),  Variable(target_transform)

        output = model(img1, img2_deformed)         

        # TODO: chagne loss here to geodesic loss
        # SE3_DIM = 6
        # weight = np.ones(SE3_DIM)
        # loss = SE3GeodesicLoss(weight)(output, target_transform)
        loss = F.mse_loss(output, target_transform)        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()               
        losses.append(loss.detach().cpu().numpy())        

    ## save the loss using tensorboardX
    writer.add_scalars('data/train', { 'loss': np.mean(losses)                            
                            }, epoch)

"""
save network weights in given path
"""
def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    # if is_best:
    #     shutil.copyfile(filename, 'model_best.pth.tar')

def main():  
    
    ## read input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=2)
    parser.add_argument('--nEpochs', type=int, default=200)
    # parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
    #                     help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--dataFolderPath', default='/data/RDRL/images/RESECT_RegistrationFiles/images/{0}/', type=str, metavar='DIR',
                        help='path to data folder')    
    parser.add_argument('--exp-name', default='', type=str,
                        help='experiment name to see later in tensorboard')
    parser.add_argument('--nBatchInEpoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    parser.add_argument('--weight-decay', '--wd', default=1e-7, type=float,
                        metavar='W', help='weight decay (default: 1e-7)')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='adam',
                        choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    print('Is cuda available : {}'.format(torch.cuda.is_available()))
    print(args)
    file_dir = os.path.dirname(os.path.realpath(__file__))    
    
    ## initialize tensorboard
    if os.environ.get('POLYAXON_RUN_OUTPUTS_PATH'):        
        if os.path.isdir(os.environ.get('POLYAXON_RUN_OUTPUTS_PATH') + '/tensorboard/'):
            shutil.rmtree(os.environ.get('POLYAXON_RUN_OUTPUTS_PATH') + '/tensorboard/')
        writer = SummaryWriter(log_dir = os.environ.get('POLYAXON_RUN_OUTPUTS_PATH') + '/tensorboard/')
    else:
        save_dir = file_dir
        if not os.path.isdir(save_dir + '/tensorboard'):     
            os.mkdir(save_dir + '/tensorboard')
        if os.path.isdir(save_dir + '/tensorboard/' + args.exp_name):
            shutil.rmtree(save_dir + '/tensorboard/' + args.exp_name)                
        writer = SummaryWriter(log_dir = save_dir + '/tensorboard/' + args.exp_name)

    ## create model
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    model = PyramidLSTM()
    print('Total number of parameters in the model: {}'.format(model.count_parameters()))

    ## optimizer
    # add lr to input arguments
    weight_decay = args.weight_decay
    if args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.99, weight_decay=weight_decay)
    elif args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay, lr=args.lr)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), weight_decay=weight_decay, lr=args.lr)
    
    if (os.path.isfile(file_dir + "/models/checkpoint-{}.pt".format(args.exp_name))):
        args.resume = file_dir + "/models/checkpoint-{}.pt".format(args.exp_name)

    ## load weights or initialize
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            if checkpoint.get('optimizer', None) is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            args.start_epoch = 0
            print("=> no checkpoint found at '{}'".format(args.resume))
            model.apply(weights_init)
    else:
        if (os.path.isdir(file_dir + "/models/")):
            args.resume = file_dir + "/models/checkpoint-{}.pt".format(args.exp_name)
        args.start_epoch = 0
        model.apply(weights_init)
        
    if args.cuda:
        model = model.cuda()

    print('optimizer lr: {}'.format(optimizer.param_groups[0]['lr']) )
    ## dataLoader
    data_folder_path = args.dataFolderPath
    mr_template_path = data_folder_path + 'brainmask_{0}.nii.gz'
    us_template_path = data_folder_path + 'Case{0}-US-before_resampled.nii.gz'
    us_format_orig = args.dataFolderPath + 'Case{0}-US-before.nii.gz'

    params = {'batch_size': args.batchSz,
            'shuffle': True,
            'num_workers': 2}

    train_loader_rigid = RigidDataLoader(args.batchSz, mr_template_path, us_template_path, us_format_orig, [1,2,3,4,5,6,7,8], [1,2,3,4,5,6], dataset_count=10)
    training_generater = DataLoader(train_loader_rigid, **params)

    test_loader_rigid = RigidDataLoader(1, mr_template_path, us_template_path, us_format_orig, [1,2,3,4,5,6,7,8], [7,8], dataset_count=2)
    test_generater = DataLoader(test_loader_rigid, **params)

    # eval(args, args.start_epoch, model, test_generater, writer)
    ## train or inference
    for epoch in range(args.start_epoch, args.nEpochs + args.start_epoch + 1):
        train(args, epoch, model, training_generater, optimizer, writer)
        if np.mod(epoch,10) == 0:
            eval(args, epoch, model, test_generater, writer)
        if np.mod(epoch,50) == 0:
            if args.resume:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),                    
                    'optimizer' : optimizer.state_dict(),
                }, False,
                args.resume)


if __name__ == "__main__":
    main()