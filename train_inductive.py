import argparse
import collections
import numpy as np
import pandas as pd 
import datetime
from tqdm import tqdm 
from easydict import EasyDict
import os 
os.chdir('./pytorch_retinanet_master')
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from retinanet import model_inductive
from retinanet.losses import FocalLoss
from retinanet.losses_polar import PolarLoss
from retinanet.dataloader import CocoDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from retinanet.model_utils import get_vocab, get_words, get_model
from retinanet import coco_eval
import wandb
print('CUDA available: {}'.format(torch.cuda.is_available()))

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--traditional_model', help='Path to traditional model (.pt)', default=None)
    parser.add_argument('--checkpoint_path', help='Path to checkpoint', default=None)
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=5)
    parser.add_argument('--batch_size', help='Size of each batch of images', type=int, default=4)
    parser.add_argument('--log_wandb', help='on or off - Log the result or not ', default= None)

    parser = parser.parse_args(args)

    ###### args PascalVOC #####
    parser = EasyDict({'dataset': 'pascalvoc',
                       'coco_path': '../data/PascalVOC',
                       'depth': 50,
                       'epochs': 10,
                       'traditional_model': '/home/qdinh/pytorch_retinanet_master/checkpoints/pascalvoc/traditional/traditional_resnet50_9.pt',
                    #    'checkpoint_path': '/home/qdinh/pytorch_retinanet_master/checkpoints/pascalvoc/inductive/inductive_resnet50_4.pt',
                        'checkpoint_path': None,
                       'batch_size':4,
                       'log_wandb': 'on',
                       'loss_fn': 'FL',
                       'emb_type': 'w2v',
                       'learning_rate': 1e-5})
        
    #### Create the data loaders ####
    if parser.dataset == 'dota':
        train_setname = 'train2017zsd'
        valseen_setname = 'val2017zsd'
        valunseen_setname = 'val2017zsd'

    elif parser.dataset == 'pascalvoc':
        train_setname = 'train0712zsd'
        valseen_setname = 'testseen2007zsd'
        valunseen_setname = 'testunseen2007zsd'

    elif parser.dataset == 'coco':
        train_setname = 'train2014seen'
        valseen_setname = 'val2014seenmini'
        valunseen_setname = 'val2014unseen'

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')
    
    dataset_train = CocoDataset(parser.coco_path, set_name=train_setname, is_zsd=True,
                                transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    dataset_valseen = CocoDataset(parser.coco_path, set_name=valseen_setname, is_zsd=True,
                                transform=transforms.Compose([Normalizer(), Resizer()]))
    dataset_valunseen = CocoDataset(parser.coco_path, set_name=valunseen_setname, is_zsd=True,
                                transform=transforms.Compose([Normalizer(), Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    #### Create the model ####
    if parser.dataset == 'dota':
        s, u = 13, 3
    elif parser.dataset == 'pascalvoc':
        s, u = 16, 4
    elif parser.dataset == 'coco':
        s, u = 65, 15
    n_reg, n_cls = s+u, s

    class_embeddings = get_words(parser.dataset, parser.emb_type)

    if parser.depth == 18:
        retinanet = model_inductive.RetinaNet18(class_embeddings, num_classes_reg_head = n_reg, num_classes_cls_head = n_cls, pretrained=True)
    elif parser.depth == 34:
        retinanet = model_inductive.RetinaNet34(class_embeddings, num_classes_reg_head = n_reg, num_classes_cls_head = n_cls, pretrained=True)
    elif parser.depth == 50:
        retinanet = model_inductive.RetinaNet50(class_embeddings, num_classes_reg_head = n_reg, num_classes_cls_head = n_cls, pretrained=True)
    elif parser.depth == 101:
        retinanet = model_inductive.RetinaNet101(class_embeddings, num_classes_reg_head = n_reg, num_classes_cls_head = n_cls, pretrained=True)
    elif parser.depth == 152:
        retinanet = model_inductive.RetinaNet152(class_embeddings, num_classes_reg_head = n_reg, num_classes_cls_head = n_cls, pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    if parser.checkpoint_path is not None: # checkpoint 
        # checkpoint model dict
        checkpoint_dict = torch.load(parser.checkpoint_path).state_dict()
        checkpoint_dict = {k: v for k, v in checkpoint_dict.items()}
        # load checkpoint
        model_dict = retinanet.state_dict()
        for layer in checkpoint_dict.keys():
            if layer in model_dict.keys():
                model_dict[layer] = checkpoint_dict[layer]
            else:
                print(layer, 'checkpoints params not loaded.')
        retinanet.load_state_dict(model_dict)
        print('Checkpoint loaded!')
    else:
        # pretrained model dict
        pretrained_dict = torch.load(parser.traditional_model).state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()}
        # load pretrained
        model_dict = retinanet.state_dict()
        for layer in pretrained_dict.keys():
            if layer in model_dict.keys():
                model_dict[layer] = pretrained_dict[layer]
            else:
                print(layer, 'pretrained params not loaded.')
        retinanet.load_state_dict(model_dict)
        print('Pretrained loaded!')
    
    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()    
                
    #### optimizers ####
    optimizer = optim.Adam(retinanet.parameters(), lr=parser.learning_rate)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    if parser.loss_fn == 'FL':
        loss_fn = FocalLoss()
    elif parser.loss_fn == 'PL':
        loss_fn = PolarLoss()

    loss_hist = collections.deque(maxlen=500)

    breakpoint()
    if parser.log_wandb == 'on':
        wandb.login()
        wandb.init(
            # set the wandb project where this run will be logged
            project = "Inductive_Word-PascalVOC",
            id      = f"word_{parser.loss_fn}",
            resume  = "allow",
            # track hyperparameters and run metadata
            config={
                "loss_fn": parser.loss_fn,
                "batch_size": parser.batch_size,
                "learning_rate": parser.learning_rate,
                "dataset": parser.dataset,
                "epochs": parser.epochs,
                }
            )

    if parser.checkpoint_path is None:
        start_epoch = 0
        print('Start training...')
    else:
        start_epoch = int(parser.checkpoint_path.split('_')[-1].strip('.pt')) + 1
        print(f'Resume from epoch num {start_epoch}...')

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
                    classifications, regressions, anchors = retinanet(data['img'].cuda().float())
                    annotations = data['annot'].cuda()
                else:
                    classifications, regressions, anchors = retinanet(data['img'].float())
                    annotations = data['annot']
                
                classification_loss, regression_loss = loss_fn(classifications, regressions, anchors, annotations)
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
                print(e)
                breakpoint()
        scheduler.step(np.mean(epoch_loss))
        if parser.log_wandb == 'on':
            wandb.log({'Epoch_total_loss': float(np.mean(epoch_loss))})
        os.makedirs(f'checkpoints/{parser.dataset}/inductive', exist_ok=True)
        torch.save(retinanet.module, f'checkpoints/{parser.dataset}/inductive/inductive_resnet{parser.depth}_{epoch_num}.pt')
        
        if ((epoch_num + 1) % 3) == 0 or (epoch_num == 0):
            ## evaluation traditional
            os.makedirs(f'log/{parser.dataset}', exist_ok=True)
            date = str(datetime.datetime.date(datetime.datetime.today()))
            print('Evaluating dataset')
            
            retinanet.eval()
            detect_type = 'traditional'
            print(f'Valset: {valseen_setname} | Epoch {epoch_num} | DetectType: {detect_type}')
            stats_seen, class_aps_seen = coco_eval.evaluate_coco(parser.dataset, dataset_valseen, retinanet, emb_type = parser.emb_type, detect_type = detect_type, return_classap=True)
            lines= [
                f'Date: Date: {date} | Dataset: {parser.dataset} | Valset: {valseen_setname} | Epoch {epoch_num} | DetectType: {detect_type} \n'
                f'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {stats_seen[0]}\n'
                f'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {stats_seen[1]}\n'
                f'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {stats_seen[2]}\n'
                # f'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {stats_seen[3]}\n'
                # f'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {stats_seen[4]}\n'
                # f'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {stats_seen[5]}\n'
                # f'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {stats_seen[6]}\n'
                # f'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {stats_seen[7]}\n'
                f'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {stats_seen[8]}\n'
                # f'Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {stats_seen[9]}\n'
                # f'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {stats_seen[10]}\n'
                # f'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {stats_seen[11]}\n'
            ]

            detect_type = 'zsd'
            print(f'Valset: {valseen_setname} | Epoch {epoch_num} | DetectType: {detect_type}')
            stats_unseen, class_aps_unseen = coco_eval.evaluate_coco(parser.dataset, dataset_valunseen, retinanet, emb_type = parser.emb_type, detect_type = detect_type, return_classap=True)
            lines += [
                f'Date: Date: {date} | Dataset: {parser.dataset} | Valset: {valunseen_setname} | Epoch {epoch_num} | DetectType: {detect_type} \n'
                f'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {stats_unseen[0]}\n'
                f'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {stats_unseen[1]}\n'
                f'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {stats_unseen[2]}\n'
                # f'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {stats_unseen[3]}\n'
                # f'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {stats_unseen[4]}\n'
                # f'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {stats_unseen[5]}\n'
                # f'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {stats_unseen[6]}\n'
                # f'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {stats_unseen[7]}\n'
                f'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {stats_unseen[8]}\n'
                # f'Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {stats_unseen[9]}\n'
                # f'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {stats_unseen[10]}\n'
                # f'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {stats_unseen[11]}\n'
            ]

            f = open(f'log/{parser.dataset}/inductive_mAPs.txt', 'a')
            f.writelines(lines)
            f.close()
            
            if parser.dataset == 'pascalvoc':
                unseen_classes = ['car', 'dog', 'sofa', 'train']
            elif parser.dataset == 'dota':
                unseen_classes = ['storage-tank', 'tennis-court', 'swimming-pool']
            if parser.dataset == 'coco':
                pass          
            seen_classes = [x for x in class_aps_seen.keys() if x not in unseen_classes]
            columns = ['Date', 'Dataset', 'SetName', 'DetectType', 'EmbType', 'Epoch', 
                       'Loss', 'Alpha', 'Gamma', 'Beta', 'mAP@0.5_seen', 'mAP@0.5_unseen', 'R@100_seen', 'R@100_unseen'] \
                        + seen_classes + unseen_classes
            if os.path.exists(f'log/{parser.dataset}/inductive_classAPs.csv'):
                f = pd.read_csv(f'log/{parser.dataset}/inductive_classAPs.csv')
            else:
                f = pd.DataFrame(columns = columns, index=None)
            try:
                line = [date, parser.dataset,  valseen_setname, 'trad/zsd', parser.emb_type, epoch_num, 
                        parser.loss_fn, loss_fn.alpha, loss_fn.gamma, loss_fn.beta,
                        stats_seen[1], stats_unseen[1], stats_seen[8], stats_unseen[8]]
            except:
                line = [date, parser.dataset,  valseen_setname, 'trad/zsd', parser.emb_type, epoch_num, 
                        parser.loss_fn, loss_fn.alpha, loss_fn.gamma, np.nan, 
                        stats_seen[1], stats_unseen[1], stats_seen[8], stats_unseen[8]]
            seen_aps = [np.round(v, 3) for (k, v) in class_aps_seen.items() if k in seen_classes]
            unseen_aps = [np.round(v, 3) for (k, v) in class_aps_unseen.items() if k in unseen_classes]
            line += seen_aps
            line += unseen_aps
            f = pd.concat([f, pd.DataFrame([line], columns=columns)])
            f.to_csv(f'log/{parser.dataset}/inductive_classAPs.csv', index = None)
            if parser.log_wandb == 'on':
                wandb.log({'mAP@0.5_seen': stats_seen[1],
                        'mAP@0.5_unseen':stats_unseen[1]})


if __name__ == '__main__':
    main()
