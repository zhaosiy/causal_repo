import csv
import time
import argparse
import pickle
import os
import datetime
import json
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import nn
from utils import *
from modules import *
import numpy as np
import matplotlib.pyplot as plt

modelpath = '/Users/siyanzhao/Desktop/causalcity/causalcity.github.io/code/NRI/nuscene/sdd_model.pt'

import pickle

with open("sdd_motion_data.pkl", "rb") as fp:
    multi_data = pickle.load(fp)
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--encoder-hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--decoder-hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--temp', type=float, default=0.5,
                    help='Temperature for Gumbel softmax.')
parser.add_argument('--encoder', type=str, default='mlp',
                    help='Type of path encoder model (mlp or cnn).')
parser.add_argument('--decoder', type=str, default='rnn',
                    help='Type of decoder model (mlp, rnn, or sim).')
parser.add_argument('--no-factor', action='store_true', default=False,
                    help='Disables factor graph model.')
parser.add_argument('--suffix', type=str, default='',
                    help='Suffix for training data (e.g. "_charged".')
parser.add_argument('--encoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--decoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('-s', '--save-folder', type=str, default='logs',
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--exp-name', type=str, default='exp_final')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')
parser.add_argument('--edge-types', type=int, default=2,
                    help='The number of edge types to infer.')
parser.add_argument('--dims', type=int, default=2,
                    help='The number of input dimensions (position + velocity).')
parser.add_argument('--timesteps', type=int, default=8,
                    help='The number of time steps per sample.')
parser.add_argument('--prediction_steps', type=int, default=12, metavar='N',
                    help='Num steps to predict before re-using teacher forcing.')
parser.add_argument('--lr-decay', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument('--skip-first', action='store_true', default=True,
                    help='Skip first edge type in decoder, i.e. it represents no-edge.')
parser.add_argument('--var', type=float, default=5e-5,
                    help='Output variance.')
parser.add_argument('--hard', action='store_true', default=False,
                    help='Uses discrete samples in training forward pass.')
parser.add_argument('--prior', nargs='+', type=float, default=[0.5, 0.5])
parser.add_argument('--dynamic-graph', action='store_true', default=False,
                    help='Whether test with dynamically re-computed graph.')
parser.add_argument('--log_per_iter', type=int, default=10)

args = parser.parse_args()
args.factor = not args.no_factor
encoder = MLPEncoder(args.timesteps * args.dims, args.encoder_hidden,
                     args.edge_types,
                     args.encoder_dropout, args.factor)
decoder = RNNDecoder(n_in_node=2,
                     edge_types=args.edge_types,
                     n_hid=args.decoder_hidden,
                     do_prob=args.decoder_dropout,
                     skip_first=args.skip_first)
eval_scenes = np.arange(125, 140, 1)  # TO DO: change this to more scenes later
print(eval_scenes)
for eval_sceneid in eval_scenes:
    directory = 'sdd_no_egde_predictions/'+str(eval_sceneid)
    if not os.path.exists(directory):
        os.makedirs(directory)
# load models:
checkpoint = torch.load(modelpath)
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])
encoder.eval()
decoder.eval()
for scene_id in eval_scenes:
    data = multi_data[str(scene_id)]
    nbr_car = data.shape[0]
    feats = data[:, :args.timesteps, :]  # nbr, 8, 2 (3.2s)
    targets = data[:, -args.prediction_steps:, :]  # nbr, 12, 2 (4.8s)
    feat_val = torch.FloatTensor(feats).unsqueeze(0)

    # Generate off-diagonal interaction graph
    args.num_atoms = feat_val.shape[1]
    off_diag = np.ones([args.num_atoms, args.num_atoms]) - np.eye(args.num_atoms)

    rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)

    rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)

    rel_rec = torch.FloatTensor(rel_rec)
    rel_send = torch.FloatTensor(rel_send)
    updated_nodes = encoder(feat_val, rel_rec, rel_send)
    output = decoder(feat_val, args.prediction_steps, updated_nodes, rel_rec, rel_send, 1).detach().numpy()
    # print(output[0][0][2], targets_val[0][0][2])
    for agent in range(nbr_car):

        X = output[0][agent][:, 0]
        Y = output[0][agent][:, 1]
        X_gt = targets[agent][:, 0]
        Y_gt = targets[agent][:, 1]
        his_X = feats[agent][:, 0]
        his_Y = feats[agent][:, 1]
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter(X, Y, s=10, c='r', marker="o", label='predict by Meshnet')
        ax1.scatter(X_gt, Y_gt, s=10, c='b', marker="s", label='ground truth')
        ax1.scatter(his_X, his_Y, s=10, c='y', marker="s", label='history')
        plt.legend(loc='upper left')
        fig.savefig('sdd_no_egde_predictions/'+str(scene_id)+'/agent_'+str(agent)+'.png')
        plt.close()
