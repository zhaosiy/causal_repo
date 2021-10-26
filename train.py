import  csv
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
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
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
parser.add_argument('--save-folder', type=str, default='logs',
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--exp-name', type=str, default='exp_final')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')
parser.add_argument('--edge-types', type=int, default=2,
                    help='The number of edge types to infer.')
parser.add_argument('--dims', type=int, default=3,
                    help='The number of input dimensions (position + velocity).')
parser.add_argument('--timesteps', type=int, default=5,
                    help='The number of time steps per sample.')
parser.add_argument('--prediction-steps', type=int, default=10, metavar='N',
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
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.factor = not args.no_factor
print(args)
writer = SummaryWriter('{}/{}/'.format(args.save_folder, args.exp_name))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Save model and meta-data. Always saves in a new sub-folder.
if args.save_folder:
    exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = '{}/{}/'.format(args.save_folder, args.exp_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    encoder_file = os.path.join(save_folder, 'encoder.pt')
    decoder_file = os.path.join(save_folder, 'decoder.pt')

    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')

    pickle.dump({'args': args}, open(meta_file, "wb"))
else:
    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")

import pickle
with open("motion_data.pkl", "rb") as fp:
    multi_data = pickle.load(fp)
print(multi_data.keys)

encoder = MLPEncoder(args.timesteps * args.dims, args.encoder_hidden,
                         args.edge_types,
                         args.encoder_dropout, args.factor)
decoder = RNNDecoder(n_in_node=2,
                         edge_types=args.edge_types,
                         n_hid=args.decoder_hidden,
                         do_prob=args.decoder_dropout,
                         skip_first=args.skip_first)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                       lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)

def train(epoch, best_val_loss):
    t = time.time()
    nll_train = []
    kl_train = []
    mse_train = []
    encoder.train()
    decoder.train()
    scheduler.step()
    loss = nn.MSELoss()
    for batch_idx in range(3):
        # batch by scenes
        random_two_scene = np.random.randint(0, 5, 2) # TO DO: change this to more scenes later

        for scene_id in random_two_scene:
            data = multi_data[str(scene_id)]
            nbr_car = data['nbr_car']
            # process data
            train_step = 5
            predict_step = 35
            feats = np.zeros((1, nbr_car, train_step, 3))
            targets = np.zeros((1, nbr_car, predict_step, 2))

            mask_matrix = np.zeros((nbr_car, 35))
            for car_id in range(nbr_car):
                all = np.concatenate([data[str(car_id)]['trans'],data[str(car_id)]['velocity']], axis=-1) # 40, 3
                mask = data[str(car_id)]['mask'][:,6:][0] # 1, 41]
                mask_matrix[car_id] = mask
                feats[0][car_id] = all[:train_step]
                targets[0][car_id] = all[train_step:,:2]
            orig_mask = mask_matrix
            feat_train = torch.FloatTensor(feats)
            targets_train = torch.FloatTensor(targets)
            mask_matrix = torch.FloatTensor(mask_matrix)
            # Generate off-diagonal interaction graph
            args.num_atoms = feat_train.shape[1]
            off_diag = np.ones([args.num_atoms, args.num_atoms]) - np.eye(args.num_atoms)

            rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)

            rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)

            rel_rec = torch.FloatTensor(rel_rec)
            rel_send = torch.FloatTensor(rel_send)
            optimizer.zero_grad()
            #print(feat_train, 'feature')
            updated_nodes = encoder(feat_train, rel_rec, rel_send)

            output = decoder(feat_train, train_step, updated_nodes, rel_rec, rel_send, 1)
            mask_matrix = mask_matrix.unsqueeze(-1).repeat(1, 1, 2)
            #print(output.shape,'output',targets_train.shape, mask_matrix.shape)
            output = output * mask_matrix
            targets_train = targets_train * mask_matrix
            print(output[0][0][25], 'output', targets_train[0][0][25], orig_mask[0][25])
            loss_mse = loss(targets_train, output)
            loss_mse.backward()

            optimizer.step()
            if batch_idx % args.log_per_iter == 0:
                writer.add_scalar("Loss/mse_train", loss_mse.item(),epoch )

            print('Epoch:', epoch, 'mse loss', loss_mse.mean())
    # evaluation
    encoder.eval()
    decoder.eval()
    if epoch % 5 == 0:
        scenes = [5, 6,7]
        val_loss = []
        for scene_id in scenes:
            data = multi_data[str(scene_id)]
            nbr_car = data['nbr_car']
            # process data
            train_step = 5
            predict_step = 35
            feats = np.zeros((1, nbr_car, train_step, 3))
            targets = np.zeros((1, nbr_car, predict_step, 2))

            mask_matrix = np.zeros((nbr_car, 35))
            for car_id in range(nbr_car):
                all = np.concatenate([data[str(car_id)]['trans'], data[str(car_id)]['velocity']], axis=-1)  # 40, 3
                mask = data[str(car_id)]['mask'][:, 6:][0]  # 1, 41]
                mask_matrix[car_id] = mask
                feats[0][car_id] = all[:train_step]
                targets[0][car_id] = all[train_step:, :2]
            orig_mask = mask_matrix
            feat_train = torch.FloatTensor(feats)
            targets_train = torch.FloatTensor(targets)
            mask_matrix = torch.FloatTensor(mask_matrix)
            # Generate off-diagonal interaction graph
            args.num_atoms = feat_train.shape[1]
            off_diag = np.ones([args.num_atoms, args.num_atoms]) - np.eye(args.num_atoms)

            rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)

            rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)

            rel_rec = torch.FloatTensor(rel_rec)
            rel_send = torch.FloatTensor(rel_send)

            # print(feat_train, 'feature')
            updated_nodes = encoder(feat_train, rel_rec, rel_send)
            output = decoder(feat_train, train_step, updated_nodes, rel_rec, rel_send, 1)
            mask_matrix = mask_matrix.unsqueeze(-1).repeat(1, 1, 2)
            # print(output.shape,'output',targets_train.shape, mask_matrix.shape)
            output = output * mask_matrix
            targets_train = targets_train * mask_matrix
            val_loss_mse = loss(targets_train, output)
            val_loss.append(val_loss_mse)
        writer.add_scalar("Val_Loss/mse_val", val_loss_mse.mean().item(), epoch)
    return loss_mse.mean()


# Train model
t_total = time.time()
best_val_loss = np.inf
best_epoch = 0
for epoch in range(args.epochs):
    val_loss = train(epoch, best_val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
print("Optimization Finished!")
print("Best Epoch: {:04d}".format(best_epoch))
if args.save_folder:
    print("Best Epoch: {:04d}".format(best_epoch), file=log)
    log.flush()