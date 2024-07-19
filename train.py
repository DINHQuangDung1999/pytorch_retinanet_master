import argparse
import collections
import datetime
import numpy as np
import json
from tqdm import tqdm 
import wandb
import os 
os.chdir('./pytorch_retinanet_master')
from easydict import EasyDict
import pandas as pd 
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from retinanet import model
from retinanet.losses import FocalLoss
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from retinanet import coco_eval
from retinanet import csv_eval
from pycocotools.cocoeval import COCOeval
print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--checkpoint_path', help='Path checkpoint', default=None)
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=30)
    parser.add_argument('--batch_size', help='Size of each batch of images', type=int, default=4)

    parser = parser.parse_args(args)
    
    #### args DOTA ###
    # parser = EasyDict({'dataset': 'dota',
    #                    'coco_path': '../data/DOTA',
    #                    'depth': 50,
    #                    'epochs': 12,
    #                    'checkpoint_path': None,
    #                     # 'checkpoint_path':'/home/qdinh/pytorch_retinanet_master/checkpoints/dota/traditional/traditional_resnet50_1.pt',
    #                    'batch_size': 4,
    #                    'log_wandb':'on'})
    #### args COCO ###
    parser = EasyDict({'dataset': 'coco',
                       'coco_path': '../data/MSCOCO',
                       'depth': 50,
                       'epochs': 30,
                       'checkpoint_path': None,
                        # 'checkpoint_path':'/home/qdinh/pytorch_retinanet_master/checkpoints/coco/traditional/traditional_resnet50_9.pt',
                       'batch_size': 4,
                       'log_wandb':'off',
                       'learning_rate': 1e-5,
                       'freeze_backbone': False})
    # #### args Pascal ###
    # parser = EasyDict({'dataset': 'pascalvoc',
    #                    'coco_path': '../data/PascalVOC',
    #                    'depth': 50,
    #                    'epochs': 30,
    #                    'checkpoint_path': None,
    #                     # 'checkpoint_path':'/home/qdinh/pytorch_retinanet_master/checkpoints/coco/traditional/traditional_resnet50_0.pt',
    #                    'batch_size': 4,
    #                    'log_wandb':'on',
    #                    'learning_rate': 1e-5,
    #                    'freeze_backbone': True})        
    #### Create the data loaders ####
    if parser.dataset == 'dota':
        train_setname = 'train2017seen'
        val_setname = 'test2017seen'

    elif parser.dataset == 'pascalvoc':
        train_setname = 'train0712zsd'
        val_setname = 'testseen2007zsd'

    elif parser.dataset == 'coco':
        train_setname = 'train2014seen'
        val_setname = 'val2014seenmini'

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')
    
    dataset_train = CocoDataset(parser.coco_path, set_name= train_setname, is_zsd=True, # load only seen classes
                                transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    dataset_val = CocoDataset(parser.coco_path, set_name=val_setname, is_zsd=True, # load only seen classes
                                transform=transforms.Compose([Normalizer(), Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    #### Create the model #### 
    if parser.dataset == 'dota':
        num_classes = 13
    elif parser.dataset == 'pascalvoc':
        num_classes = 16 
    elif parser.dataset == 'coco':
        num_classes = 65

    if parser.depth == 18:
        retinanet = model.RetinaNet18(num_classes=num_classes, pretrained=True) # train with only 13 seen classes
    elif parser.depth == 34:
        retinanet = model.RetinaNet34(num_classes=num_classes, pretrained=True)
    elif parser.depth == 50:
        retinanet = model.RetinaNet50(num_classes=num_classes, pretrained=True)
    elif parser.depth == 101:
        retinanet = model.RetinaNet101(num_classes=num_classes, pretrained=True)
    elif parser.depth == 152:
        retinanet = model.RetinaNet152(num_classes=num_classes, pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda() 

    # freeze_backbone
    if parser.freeze_backbone == True:
        for k, v in retinanet.named_parameters():
            if 'classificationModel' not in k and 'regressionModel' not in k:
                v.requires_grad = False
        print('Freezed backbone!')
        
    # checkpoint 
    if parser.checkpoint_path is not None:
        checkpoint = torch.load(parser.checkpoint_path)
        retinanet.module.load_state_dict(checkpoint.state_dict())
    
    #### optimizers ####
    optimizer = optim.Adam(retinanet.parameters(), lr=parser.learning_rate)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    loss_hist = collections.deque(maxlen=500)
    
    loss_fn = FocalLoss()

    if parser.checkpoint_path is None:
        start_epoch = 0
        print('Start training...')
    else:
        start_epoch = int(parser.checkpoint_path.split('_')[-1].strip('.pt')) + 1
        print(f'Resume from epoch num {start_epoch}...')

    breakpoint()
    if parser.log_wandb == 'on':
        wandb.login()
        wandb.init(
            # set the wandb project where this run will be logged
            project = "Traditional-PascalVOC",
            id      = "traditional_0",
            resume  = "allow",
            # track hyperparameters and run metadata
            config={
                "freeze_backbone": parser.freeze_backbone,
                "batch_size": parser.batch_size,
                "learning_rate": parser.learning_rate,
                "dataset": parser.dataset,
                "epochs": parser.epochs,
                }
            )
    #### train #### 
    for epoch_num in range(start_epoch, parser.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(tqdm(dataloader_train)):
        #     break 
        # break 
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification, regression, anchors = retinanet(data['img'].cuda().float())
                else:
                    classification, regression, anchors = retinanet(data['img'].float())

                annotations = data['annot']
                if torch.cuda.is_available():
                    annotations = annotations.cuda()

                classification_loss, regression_loss = loss_fn(classification, regression, anchors, annotations)
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                
                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                _ = torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.001)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                # print(
                #     'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                #         epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))
                if parser.log_wandb == 'on':
                    wandb.log({'train_cls_loss' : float(classification_loss),
                            'train_reg_loss'  : float(regression_loss),
                            'train_total_loss': float(loss)})
                
                del classification_loss
                del regression_loss

            except Exception as e:
                raise RuntimeError(e)
            
        if parser.log_wandb == 'on':
            wandb.log({'train_epoch_total_loss': np.mean(epoch_loss)})

        os.makedirs(f'checkpoints/{parser.dataset}/traditional', exist_ok=True)
        torch.save(retinanet.module, f'checkpoints/{parser.dataset}/traditional/traditional_resnet{parser.depth}_{epoch_num}.pt')
        scheduler.step(np.mean(epoch_loss))

        ## evaluation
        if ((epoch_num + 4) % 1) == 0 or (epoch_num == 0):
            os.makedirs(f'log/{parser.dataset}', exist_ok=True)
            date = str(datetime.datetime.date(datetime.datetime.today()))
            print('Evaluating dataset')
            
            retinanet.eval()
            detect_type = 'traditional'
            stats, class_aps = coco_eval.evaluate_coco(parser.dataset, dataset_val, retinanet, detect_type = detect_type, return_classap=True)

            lines= [
                f'Date: {date} | Dataset: {parser.dataset} | Valset: {val_setname} | Epoch {epoch_num} | DetectType: {detect_type} \n'
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

            f = open(f'log/{parser.dataset}/traditional_mAPs.txt', 'a')
            f.writelines(lines)
            f.close()
            
            if parser.dataset == 'coco':  

                columns = ['Date', 'Dataset', 'SetName', 'DetectType', 'Epoch', 'Alpha', 'Gamma', 'mAP@0.5', 'mAP@0.75', 'R@100']

                if os.path.exists(f'log/{parser.dataset}/traditional_classAPs.csv'):
                    f = pd.read_csv(f'log/{parser.dataset}/traditional_classAPs.csv')
                else:
                    f = pd.DataFrame(columns = columns, index=None)

                line = [date, parser.dataset, val_setname, detect_type, epoch_num, loss_fn.alpha, loss_fn.gamma, stats[1], stats[2], stats[8]]
                f = pd.concat([f, pd.DataFrame([line], columns=columns)])
                f.to_csv(f'log/{parser.dataset}/traditional_classAPs.csv', index = None)
                if parser.log_wandb == 'on':
                    wandb.log({'mAP@0.5': stats[1], 
                               'mAP@0.75': stats[2],
                               'R@100': stats[8]})

            else:

                if parser.dataset == 'pascalvoc':
                    unseen_classes = ['car', 'dog', 'sofa', 'train']
                elif parser.dataset == 'dota':
                    unseen_classes = ['storage-tank', 'tennis-court', 'swimming-pool']

                seen_classes = [x for x in class_aps.keys() if x not in unseen_classes]
                columns = ['Date', 'Dataset', 'SetName', 'DetectType', 'Epoch', 'Alpha', 'Gamma'] + seen_classes + ['mAP@0.5', 'R@100']

                if os.path.exists(f'log/{parser.dataset}/traditional_classAPs.csv'):
                    f = pd.read_csv(f'log/{parser.dataset}/traditional_classAPs.csv')
                else:
                    f = pd.DataFrame(columns = columns, index=None)

                line = [date, parser.dataset, val_setname, detect_type, epoch_num, loss_fn.alpha, loss_fn.gamma]
                seen_aps = [np.round(v, 3) for (k, v) in class_aps.items() if k in seen_classes]
                line += seen_aps
                line += [stats[1], stats[8]]

                f = pd.concat([f, pd.DataFrame([line], columns=columns)])
                f.to_csv(f'log/{parser.dataset}/traditional_classAPs.csv', index = None)
                if parser.log_wandb == 'on':
                    wandb.log({'mAP@0.5': stats[1], 
                               'mAP@0.75': stats[2],
                               'R@100': stats[8]})


if __name__ == '__main__':
    main()
