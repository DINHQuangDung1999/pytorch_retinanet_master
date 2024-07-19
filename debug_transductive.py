import argparse
import collections

import os 
os.chdir('./pytorch_retinanet_master')

import numpy as np
import json
import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model_transductive, model_inductive
from retinanet.dataloader import CocoDataset, collater, Resizer, Normalizer

from torch.utils.data import DataLoader
from detect_img import detect
from retinanet import  coco_eval_zsd
from retinanet.losses_transductive import TransductiveLoss
from tqdm import tqdm 

import wandb
import datetime 

# assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--inductive_model', help='Path to inductive model (.pt)', default=None)
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=5)
    parser.add_argument('--batch_size', help='Size of each batch of images', type=int, default=4)
    parser.add_argument('--th', help='Dynamic loss threshold', type=float, default=0.3)
    parser = parser.parse_args(args)

    from easydict import EasyDict
    parser = EasyDict({'dataset': 'pascalvoc',
                       'coco_path': '../data/PascalVOC',
                       'depth': 50,
                       'epochs': 3,
                       'inductive_model': '/home/qdinh/pytorch_retinanet_master/checkpoints/pascalvoc/inductive/30/inductive_resnet50_29.pt',
                    #    'checkpoint_path': '/home/qdinh/pytorch_retinanet_master/checkpoints/pascalvoc/inductive/inductive_resnet50_4.pt',
                        'checkpoint_path': None,
                        'log_wandb': 'off',
                        'th': 0.3,
                       'batch_size': 4})
    
    ##########################################
    ######## Create the data loaders #########
    ##########################################

    if parser.dataset == 'dota':
        set_name = 'test2017zsd'
    elif parser.dataset == 'pascalvoc':
        set_name = 'test2007zsd'
    else:
        raise ValueError('Dataset type not understood (must be dota or pascalvoc), exiting.')
    
    dataset_val = CocoDataset(parser.coco_path, set_name=set_name, is_zsd=True, 
                              transform=transforms.Compose([Normalizer(), Resizer()]))
    dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater)

    ##########################################
    ############ Create the model ############
    ##########################################
    
    ### w2v Word embeddings
    if parser.dataset == 'dota':
        
        f = open('../data/dota_dataset/w2v_embeddings.json', 'r')
        w2v_embeddings = json.load(f)

        classes = list(w2v_embeddings.keys())
        seen_ids = [1,2,4,6,7,8,9,10,11,12,14,15,16]
        unseen_ids = [3,5,13]
        s, u = 13, 3
 
    elif parser.dataset == 'pascalvoc':

        f = open('../data/PascalVOC/w2v_embeddings.json', 'r')
        w2v_embeddings = json.load(f)

        classes = list(w2v_embeddings.keys())
        seen_ids = [1,2,3,4,5,6,8,9,10,11,13,14,15,16,17,20]
        unseen_ids = [7,12,18,19]
        s, u = 16, 4
    
    tmp = []
    for i in seen_ids:
        tmp.append(w2v_embeddings[classes[i-1]])
    for i in unseen_ids:
        tmp.append(w2v_embeddings[classes[i-1]])
    w2v_embeddings = np.array(tmp)
    from sklearn.preprocessing import normalize
    w2v_embeddings = normalize(w2v_embeddings)
    
    ##### Define model

    retinanet =  model_transductive.RetinaNet50(num_classes_reg_head=s+u, num_classes_cls_head=s, pretrained=True)
    retinanet.classificationModel._set_w2v_embeddings(w2v_embeddings)
    retinanet.regressionModel._set_w2v_embeddings(w2v_embeddings)
    # breakpoint()
    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()    

    #### Load pretrained inductive model
    model_ind =  model_inductive.RetinaNet50(num_classes_reg_head=s+u, num_classes_cls_head=s, pretrained=True)
    model_ind.load_state_dict(torch.load(parser.inductive_model).state_dict())
    # breakpoint()
    pretrained_dict = model_ind.state_dict()
    model_dict = retinanet.module.state_dict()
    for k, v in model_ind.named_parameters():
        if 'w2v_embeddings' not in k:
            model_dict[k] = pretrained_dict[k]
    retinanet.module.load_state_dict(model_dict)

    ### freeze all layers except the classification subnet
    for k, v in retinanet.named_parameters():
        if 'classificationModel' not in k:
            v.requires_grad = False
    for k, v in retinanet.named_parameters():
        print(k, v.requires_grad)
    print('Pretrained loaded!')  

    # for (k1, w1), (k2, w2) in zip(model_ind.named_parameters(), retinanet.module.named_parameters()):
    #     if 'w2v_embeddings' not in k1:
    #         print(k1, (w1-w2).sum().item())
    #     else:
    #         try:
    #             print(k1, (w1-w2).sum().item())
    #         except Exception as e:
    #             print(k1, e)
    # breakpoint()

    ##########################################
    ############ Optimizers, loss ############
    ##########################################

    optimizer = optim.Adam(retinanet.module.parameters(), lr=1e-5) # only train the cls head

    loss_hist = collections.deque(maxlen=500)

    loss_fn = TransductiveLoss(n_seen = s, th = parser.th)

    ##########################################
    ################ Training ################ 
    ##########################################
    if torch.cuda.is_available():
        model_ind = torch.nn.DataParallel(model_ind).cuda()
    model_ind.eval()
    from copy import deepcopy
    retinanet_tmp = deepcopy(retinanet)
    retinanet_tmp.eval()
    # unseen_scores = []
    for epoch_num in range(parser.epochs):

        retinanet.module.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(tqdm(dataloader_val)):
        #     break
        # break
            # breakpoint()
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classifications, regressions, anchors = retinanet(data['img'].cuda().float(), w2v_embeddings, s, False)
                else:
                    classifications, regressions, anchors = retinanet(data['img'].float(), w2v_embeddings, s, False)
                # unseen_scores.append(classifications[0,:,16:].detach())
                with torch.no_grad():
                    annotations = model_ind(data['img'].cuda().float())
                    annotations = torch.cat([annotations[2], annotations[1].unsqueeze(1)], dim = 1).unsqueeze(0)
                #     res0 = retinanet_tmp(data['img'].cuda().float(), w2v_embeddings, s, False)
                # breakpoint()
                lmda = 0.4

                fixed_loss, dynamic_seen_loss, dynamic_unseen_loss = loss_fn(classifications, anchors, annotations)

                dynamic_unseen_loss = torch.tensor([0.]).cuda()
                loss = lmda*fixed_loss + (1-lmda)*(dynamic_seen_loss + dynamic_unseen_loss)
                # loss = lmda*fixed_loss + (1-lmda)*(dynamic_seen_loss)

                if bool(loss == 0):
                    continue
                # pre_params = {}
                # for k_pre, v_pre in retinanet.module.named_parameters():
                #     pre_params[k_pre] = deepcopy(v_pre)

                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(retinanet.module.parameters(), 0.1)
    
                optimizer.step()

                # post_params = {}
                # for k_post, v_post in retinanet.module.named_parameters():
                #     post_params[k_post] = deepcopy(v_post)
                # if iter_num % 100 == 0:
                #     for key in pre_params.keys():
                #         diff = torch.sum(torch.abs(post_params[key] - pre_params[key])).item()
                #         if diff > 0:
                #             print(key, diff)
                #     breakpoint()
                
                # breakpoint()
                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iteration: {} | seen max: {:.4f}. seen min: {:.4f}. unseen max: {:.4f}. unseen min: {:.4f}. \
                        Lf: {:1.5f} | Ld_1: {:1.5f} | Ld_2: {:1.5f} | L: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, 
                        classifications[0,:,:16].max(), classifications[0,:,:16].min(), classifications[0,:,16:].max(), classifications[0,:,16:].min(),
                        float(fixed_loss), float(dynamic_seen_loss), float(dynamic_unseen_loss), float(loss),
                        np.mean(loss_hist)))

            except Exception as e:
                raise RuntimeError(e)
   
        base = parser.inductive_model.split('_')[-1].strip('.pt')
        os.makedirs(f'checkpoints/{parser.dataset}/transductive/{base}', exist_ok=True)
        torch.save(retinanet.module, f'checkpoints/{parser.dataset}/transductive/{base}/transductive_resnet{parser.depth}_{epoch_num}.pt')
    import matplotlib.pyplot as plt 
    fig = plt.figure(figsize = (6,6))
    plt.hist(np.arctanh(unseen_scores.ravel()))
    plt.yscale('log')
    plt.savefig('unseen hist.png')   
    unseen_scores=classifications[0,:,16:].detach().cpu().numpy()
    seen_scores=classifications[0,:,:16].detach()
    
    x_range = np.linspace(0,1,100)
    loss_th = []
    for th in x_range:
        loss_fn = TransductiveLoss(n_seen = s, th = th)
        fixed_loss, dynamic_seen_loss, dynamic_unseen_loss = loss_fn(classifications, anchors, annotations)   
        loss_th.append(dynamic_unseen_loss.detach().cpu().item())
    loss_th = np.array(loss_th)
    fig = plt.figure(figsize = (6,6))
    plt.plot(x_range,loss_th)
    plt.yscale('log')
    plt.savefig('dynamic unseen loss.png')   

    x_range = np.linspace(0.0000000001,1,100)
    def dynamic_loss(p, mode, th = 0.3, gamma = 1):
        if mode == 'unseen':
            if p > th:
                return -((1-p)**gamma)*np.log(p**p)
            else:
                return -(p**gamma)*np.log((1-p)**p)
        elif mode == 'seen':
            if p > th:
                return -((1-p)**gamma)*np.log(p)
            else:
                return -(p**gamma)*np.log((1-p))
    ##
    unseen_loss = []
    for th in [0.1,0.2,0.3,0.4,0.5]:
        loss_list = []
        for p in x_range:
            loss_list.append(dynamic_loss(p, 'unseen', th = th))
        unseen_loss.append(loss_list)
    fig = plt.figure(figsize = (6,6))
    for i, th in enumerate([0.1,0.2,0.3,0.4,0.5]):
        plt.plot(x_range, unseen_loss[i], label = f'th: {th}')
    plt.legend()
    plt.savefig('dynamic unseen loss th.png')   
    ##
    unseen_loss = []
    for gamma in [0, 0.5, 1, 2, 5]:
        loss_list = []
        for p in x_range:
            loss_list.append(dynamic_loss(p, 'unseen', th = 0.3, gamma = gamma))
        unseen_loss.append(loss_list)
    fig = plt.figure(figsize = (6,6))
    for i, gamma in enumerate([0, 0.5, 1, 2, 5]):
        plt.plot(x_range, unseen_loss[i], label = f'gamma: {gamma}')
    plt.legend()
    plt.savefig('dynamic unseen loss gamma.png')   
    ##
    seen_loss = []
    for th in [0.1,0.2,0.3,0.4,0.5]:
        loss_list = []
        for p in x_range:
            loss_list.append(dynamic_loss(p, 'seen', th = th))
        seen_loss.append(loss_list)
    fig = plt.figure(figsize = (6,6))
    for i, th in enumerate([0.1,0.2,0.3,0.4,0.5]):
        plt.plot(x_range, seen_loss[i], label = f'th: {th}')
    plt.legend()
    plt.savefig('dynamic seen loss th.png')   
    ##
    seen_loss = []
    for gamma in [0, 0.5, 1, 2, 5]:
        loss_list = []
        for p in x_range:
            loss_list.append(dynamic_loss(p, 'seen', th = 0.3, gamma = gamma))
        seen_loss.append(loss_list)
    fig = plt.figure(figsize = (6,6))
    for i, gamma in enumerate([0, 0.5, 1, 2, 5]):
        plt.plot(x_range, seen_loss[i], label = f'gamma: {gamma}')
    plt.legend()
    plt.savefig('dynamic seen loss gamma.png')   
    ##
    x_range = np.linspace(0.00000000001,1,100)
    def focal_loss(p, gamma):
        return -((1-p)**gamma)*np.log(p)
    _loss = []
    for gamma in [0, 0.5, 1, 2, 5]:
        loss_list = []
        for p in x_range:
            loss_list.append(focal_loss(p, gamma = gamma))
        _loss.append(loss_list)
    fig = plt.figure(figsize = (6,6))
    for i, gamma in enumerate([0, 0.5, 1, 2, 5]):
        plt.plot(x_range, _loss[i], label = f'gamma: {gamma}')
    plt.legend()
    # plt.ylim(0,5) 
    plt.savefig('focal loss.png')   

    ##########################################
    ############### Evaluation ############### 
    ##########################################
    print('End training.')
    print('Lambda: {:.2f}. Alpha: {:.2f}. Gamma: {:.2f}. Beta: {:.2f}. Eta: {:.2f}. th: {:.2f}.'.format(
           lmda, loss_fn.alpha, loss_fn.gamma, loss_fn.beta, loss_fn.eta, parser.th))
    print('Evaluating dataset')
    retinanet.eval()
    os.makedirs('log/mAP', exist_ok=True)
    stats, class_ap = coco_eval_zsd.evaluate_coco_zsd(dataset_val, retinanet, type = 'gzsd', return_classap=True)

    class_aps_lines =[
        f'{key} (AP): {np.round(class_ap[key], 3)}' for key in class_ap.keys()
    ]
    print(class_aps_lines)
if __name__ == '__main__':
    main()
