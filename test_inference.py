import argparse
import collections
import os 
os.chdir('./pytorch_retinanet_master')
from tqdm import tqdm 
import numpy as np
import pandas as pd 
import datetime
import datetime 
import wandb

import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from retinanet import model_inductive, model
from retinanet.losses_polar import PolarLoss
from retinanet.dataloader import CocoDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from retinanet import coco_eval
from retinanet.model_utils import get_words, get_vocab

path_trad = '/home/qdinh/pytorch_retinanet_master/checkpoints/pascalvoc/traditional/traditional_resnet50_29.pt'
path_ind = '/home/qdinh/pytorch_retinanet_master/checkpoints/pascalvoc/inductive/FL30/inductive_resnet50_29.pt'

dataset = CocoDataset('../data/PascalVOC', set_name='testunseen2007zsd', is_zsd=True,
                            transform=transforms.Compose([Normalizer(), Resizer()]))
dataloader_train = DataLoader(dataset, num_workers=3, collate_fn=collater)

class_embeddings= get_words('pascalvoc', 'w2v')

model_trad = torch.load(path_trad).cuda()
model_ind = torch.load(path_ind).cuda()
model_trad.eval()
model_ind.eval()

with torch.no_grad():
    for iter_num, data in enumerate(tqdm(dataloader_train)):
        print(f'Image number {iter_num}')
        inputs = data['img'].cuda().float()
        results_trad = model_trad(inputs)
        results_ind = model_ind(inputs)

        # scores_trad = unseen_inference(results_trad[0], class_embeddings, 16)[0]
        # scores_trad_max = torch.max(scores_trad, dim = 1)[0]
        # scores_trad_argmax = (scores_trad_max > 0.6).nonzero()

        # scores_ind = unseen_inference(results_ind[0], class_embeddings, 16)[0]

        scores_trad = results_trad[0][0]
        scores_trad_max = torch.max(scores_trad, dim = 1)[0]
        scores_trad_argmax = (scores_trad_max > 0.05).nonzero()

        scores_ind = results_ind[0][0]
        n_dif = 0
        for i in scores_trad_argmax.squeeze(0):
            label_trad = torch.argmax(scores_trad[i]).item()
            label_ind = torch.argmax(scores_ind[i]).item()
            if label_trad != label_ind:
                n_dif += 1
                # print(scores_trad[i])
                # print(scores_ind[i])
                # print(label_trad, label_ind)
        print(f'dif ratio: {np.round(n_dif/scores_trad.shape[0], 5)}')
        if iter_num == 100:
            break



def unseen_inference(classifications, class_embeddings, n_seen):
    #### Inferring unseen scores ####
    class_embeddings_tensor = torch.from_numpy(class_embeddings).float()
    if torch.cuda.is_available():
        class_embeddings_tensor = class_embeddings_tensor.cuda()
    emb_size = class_embeddings_tensor.shape[1]
    batch_size = classifications.shape[0]
    u_classifications = []
    word_seen = class_embeddings_tensor[:n_seen,:]
    word_unseen = class_embeddings_tensor[n_seen:,:]
    for j in range(batch_size):

        # u_cls = classifications[j, :, :]
        # u_cls = u_cls[:,:n_seen]
        # T = 5

        # mask = torch.ones_like(u_cls,dtype=torch.float32).cuda()
        # mask[:, T:] = 0.0
        # sorted_u_cls, sorted_u_cls_arg = torch.sort(-u_cls, dim=1)
        # sorted_u_cls = -sorted_u_cls
        # sorted_u_cls = sorted_u_cls * mask
        # restroed_score = mask
        # for i in range(u_cls.shape[0]):
        #     restroed_score[i, sorted_u_cls_arg[i, :]] = sorted_u_cls[i, :]

        # unseen_pd = restroed_score @ word_seen
        # unseen_scores = unseen_pd @ word_unseen.T
        # u_classifications.append(unseen_scores) 

        u_cls = classifications[j, :, :]
        u_cls = u_cls[:,:n_seen]
        
        topT_scores, topT_idx = torch.topk(u_cls, k=5, dim =-1) # p'
        W_topT = class_embeddings_tensor.repeat(u_cls.shape[0], 1, 1) # W'
        topT_idx = topT_idx.unsqueeze(-1).repeat(1,1, emb_size)
        W_topT = torch.gather(W_topT, 1, topT_idx)

        W_u = class_embeddings_tensor[n_seen:,:].repeat(u_cls.shape[0],1,1) # Wu
        u_scores = topT_scores.unsqueeze(1) @ W_topT @ W_u.permute(0,2,1)
        u_scores = u_scores.squeeze(1)
        # u_scores = torch.nn.functional.tanh(u_scores)
        u_classifications.append(u_scores) 

        # breakpoint()
    # u_classifications = torch.zeros(u_scores.unsqueeze(0).shape).cuda()
    u_classifications = torch.stack(u_classifications)

    return u_classifications