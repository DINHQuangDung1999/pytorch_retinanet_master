import argparse
import collections
import numpy as np
import datetime
from tqdm import tqdm 
import os 
os.chdir('./pytorch_retinanet_master')

import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from retinanet import model_transductive
from retinanet.losses_transductive import TransductiveLoss
from retinanet.dataloader import CocoDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from retinanet.model_utils import get_vocab, get_words, get_model
from retinanet import coco_eval
from retinanet.detect_utils import detect_from_pred_boxes, unseen_inference
import wandb
print('CUDA available: {}'.format(torch.cuda.is_available()))

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--inductive_model', help='Path to inductive model (.pt)', default=None)
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int)
    parser.add_argument('--log_wandb', help='on or off - Log the result or not ', default= None)
    parser.add_argument('--th', help='dynamic loss threshold')
    parser = parser.parse_args(args)

    from easydict import EasyDict
    parser = EasyDict({'dataset': 'pascalvoc',
                       'coco_path': '../data/PascalVOC',
                       'depth': 50,
                       'epochs': 3,
                       'inductive_model': '/home/qdinh/pytorch_retinanet_master/checkpoints/pascalvoc/inductive/inductive_resnet50_14.pt',
                    #    'checkpoint_path': '/home/qdinh/pytorch_retinanet_master/checkpoints/pascalvoc/inductive/inductive_resnet50_4.pt',
                        'checkpoint_path': None,
                        'log_wandb': 'off',
                       'lmda': 0.5,
                       'th': 0.3}
                       )
    
    ##########################################
    ######## Create the data loaders #########
    ##########################################

    if parser.dataset == 'dota':
        set_name = 'test2017zsd'
    elif parser.dataset == 'pascalvoc':
        set_name = 'testunseen2007zsd'
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
        s, u = 13, 3
    elif parser.dataset == 'pascalvoc':
        s, u = 16, 4

    ##### Define model
    if parser.depth == 18:
        retinanet = model_transductive.RetinaNet18(num_classes_reg_head=s+u, num_classes_cls_head=s, pretrained=True) # train with only 13 seen classes
    elif parser.depth == 34:
        retinanet =  model_transductive.RetinaNet34(num_classes_reg_head=s+u, num_classes_cls_head=s, pretrained=True)
    elif parser.depth == 50:
        retinanet =  model_transductive.RetinaNet50(num_classes_reg_head=s+u, num_classes_cls_head=s, pretrained=True)
    elif parser.depth == 101:
        retinanet =  model_transductive.RetinaNet101(num_classes_reg_head=s+u, num_classes_cls_head=s, pretrained=True)
    elif parser.depth == 152:
        retinanet =  model_transductive.RetinaNet152(num_classes_reg_head=s+u, num_classes_cls_head=s, pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    
    class_embeddings = get_words(parser.dataset)
    retinanet.classificationModel._set_embeddings(class_embeddings)
    retinanet.regressionModel._set_embeddings(class_embeddings)

    #### Load pretrained inductive model
    model_ind = torch.load(parser.inductive_model)
    # # breakpoint()
    pretrained_dict = model_ind.state_dict()
    # model_dict = retinanet.state_dict()
    # for k, v in model_ind.named_parameters():
    #     if 'embeddings' not in k:
    #         model_dict[k] = pretrained_dict[k]
    # retinanet.load_state_dict(model_dict)
    retinanet.load_state_dict(pretrained_dict)
    ### freeze all layers except the classification subnet
    for k, v in retinanet.named_parameters():
        if 'classificationModel' not in k:
            v.requires_grad = False
    print('Trainable layers:')  
    for k, v in retinanet.named_parameters():
        if v.requires_grad == True:
            print(k, v.requires_grad)
    
    # breakpoint()
    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()

    ##########################################
    ############ Optimizers, loss ############
    ##########################################

    optimizer = optim.Adam(retinanet.module.parameters(), lr=1e-5)

    loss_hist = collections.deque(maxlen=500)

    loss_fn = TransductiveLoss(n_seen = s, th = parser.th, eta = 2, beta = 0.25)

    if parser.log_wandb == 'on':
        wandb.login()
        wandb.init(
            # set the wandb project where this run will be logged
            project = "Transductive",
            id      = "transductive",
            resume  = "allow",
            # track hyperparameters and run metadata
            config={
            "learning_rate": 1e-5,
            "depth" : parser.depth,
            "dataset": parser.dataset,
            "epochs": parser.epochs
            }
        )

    ##########################################
    ################ Training ################ 
    ##########################################
    if torch.cuda.is_available():
        model_ind = torch.nn.DataParallel(model_ind).cuda()
    model_ind.eval()

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
                    classifications, regressions, anchors = retinanet(data['img'].cuda().float())
                else:
                    classifications, regressions, anchors = retinanet(data['img'].float())
                # unseen inference
                u_classifications = unseen_inference(classifications, class_embeddings, s)
                classifications = classifications[:,:,:s]
                classifications = torch.cat([classifications, u_classifications], dim = -1)

                with torch.no_grad():
                    if torch.cuda.is_available():
                        pseudo_classifications, pseudo_regressions, pseudo_anchors = model_ind(data['img'].cuda().float())
                    else:
                        pseudo_classifications, pseudo_regressions, pseudo_anchors = model_ind(data['img'].float())
                    pseudo_scores, pseudo_labels, pseudo_boxes = detect_from_pred_boxes(data['img'].cuda().float(), 
                                                                   pseudo_classifications, pseudo_regressions, pseudo_anchors,
                                                                   detect_type = 'traditional', n_seen = s, class_embeddings= None, 
                                                                   nboxes=None)
                    pseudo_annotations = torch.cat([pseudo_boxes, pseudo_labels.unsqueeze(1)], dim = 1).unsqueeze(0)

                    # from copy import deepcopy
                    # retinanet_tmp = deepcopy(retinanet)
                    # retinanet_tmp.eval()
                    # res0 = retinanet_tmp(data['img'].cuda().float(), class_embeddings, s, False)
                    # for c in torch.unique(res0[1]):
                    #     try:
                    #         print((annotations[annotations[:,:,4] == c][:,:4] - res0[2][res0[1] == c]).sum())
                    #     except:
                    #         print(c, 'error')
                    # breakpoint()

                fixed_loss, dynamic_seen_loss, dynamic_unseen_loss = loss_fn(classifications, anchors, pseudo_annotations)

                # dynamic_unseen_loss = torch.tensor([0.]).cuda()
                # dynamic_seen_loss = torch.tensor([0.]).cuda()
                loss = parser.lmda*fixed_loss + (1-parser.lmda)*(dynamic_seen_loss + dynamic_unseen_loss)

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
                        classifications[0,:,:s].max(), classifications[0,:,:s].min(), classifications[0,:,s:].max(), classifications[0,:,s:].min(),
                        float(fixed_loss), float(dynamic_seen_loss), float(dynamic_unseen_loss), float(loss),
                        np.mean(loss_hist)))
                # breakpoint()
            except Exception as e:
                raise RuntimeError(e)
             
        base = parser.inductive_model.split('_')[-1].strip('.pt')
        os.makedirs(f'checkpoints/{parser.dataset}/transductive', exist_ok=True)
        torch.save(retinanet.module, f'checkpoints/{parser.dataset}/transductive/transductive_resnet{parser.depth}_{epoch_num}.pt')
       
        ##########################################
        ############### Evaluation ############### 
        ##########################################
        os.makedirs(f'log/{parser.dataset}', exist_ok=True)
        date = str(datetime.datetime.date(datetime.datetime.today()))
        print('Evaluating dataset')
        
        retinanet.eval()
        detect_type = 'zsd'
        stats, class_aps = coco_eval.evaluate_coco(parser.dataset, dataset_val, retinanet, threshold = 0.3, detect_type = detect_type, return_classap=True)
        # classification, regression, anchors = retinanet(data['img'].cuda().float())
        lines= [
            f'Date: {date} | Dataset: {parser.dataset} | Valset: {set_name} | Epoch {epoch_num} | DetectType: {detect_type} \n'
            f'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {stats[0]}\n'
            f'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {stats[1]}\n'
            f'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {stats[2]}\n'
            # f'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {stats[3]}\n'
            # f'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {stats[4]}\n'
            # f'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {stats[5]}\n'
            # f'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {stats[6]}\n'
            # f'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {stats[7]}\n'
            f'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {stats[8]}\n'
            # f'Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {stats[9]}\n'
            # f'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {stats[10]}\n'
            # f'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {stats[11]}\n'
        ]
        
        f = open(f'log/{parser.dataset}/transductive_mAPs.txt', 'a')
        f.writelines(lines)
        f.close()
        
        import pandas as pd 
        unseen_classes = ['car', 'dog', 'sofa', 'train']
        seen_classes = [x for x in class_aps.keys() if x not in ['car', 'dog', 'sofa', 'train']]
        columns = ['Date', 'Dataset', 'SetName', 'DetectType', 'Ind_epoch', 'Trans_epoch', 'Lambda', 'Alpha', 'Gamma', 'Beta', 'Eta', 'dynloss_th'] \
            + seen_classes + unseen_classes
        
        if os.path.exists(f'log/{parser.dataset}/transductive_classAPs.csv'):
            f = pd.read_csv(f'log/{parser.dataset}/transductive_classAPs.csv')
        else:
            f = pd.DataFrame(columns = columns, index=None)

        line = [date, parser.dataset, set_name, detect_type, base, epoch_num, parser.lmda, loss_fn.alpha, loss_fn.gamma, loss_fn.beta, loss_fn.eta, loss_fn.th]
        seen_aps = [np.round(v, 3) for (k, v) in class_aps.items() if k not in ['car', 'dog', 'sofa', 'train']]
        unseen_aps = [np.round(v, 3) for (k, v) in class_aps.items() if k in ['car', 'dog', 'sofa', 'train']]
        line += seen_aps
        line += unseen_aps

        f = pd.concat([f, pd.DataFrame([line], columns=columns)])
        f.to_csv(f'log/{parser.dataset}/transductive_classAPs.csv', index = None)

        if parser.log_wandb == 'on':
            wandb.log({'mAP_seen': np.mean(seen_aps),
                       'mAP_unseen':np.mean(unseen_aps)})



if __name__ == '__main__':
    main()
