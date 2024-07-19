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

from retinanet import model_polar
from retinanet.losses_polar import PolarLoss
from retinanet.dataloader import CocoDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from retinanet import coco_eval
from retinanet.model_utils import get_words, get_vocab

print('CUDA available: {}'.format(torch.cuda.is_available()))

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--checkpoint_path', help='Path checkpoint', default=None)
    parser.add_argument('--traditional_model', help='Path to pretrained model', default=None)
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--batch_size', help='Size of each batch of images', type=int, default=4)

    parser = parser.parse_args(args)
    
    #### args PascalVOC ###
    from easydict import EasyDict
    parser = EasyDict({'dataset': 'pascalvoc',
                       'coco_path': '../data/PascalVOC',
                       'depth': 50,
                       'epochs': 30,
                    #    'checkpoint_path': None,
                       'checkpoint_path': '/home/qdinh/pytorch_retinanet_master/checkpoints/pascalvoc/polar/w2v/polar_resnet50_24.pt',
                       'traditional_model':'/home/qdinh/pytorch_retinanet_master/checkpoints/pascalvoc/traditional/traditional_resnet50_29.pt',
                       'batch_size': 4,
                       'log_wandb': 'off',
                       'emb_type': 'w2v'})
    
    # #### args DOTA###
    # from easydict import EasyDict
    # parser = EasyDict({'dataset': 'dota',
    #                    'coco_path': '../data/DOTA',
    #                    'depth': 50,
    #                    'epochs': 10,
    #                    'checkpoint_path': None,
    #                    'traditional_model':'/home/qdinh/pytorch_retinanet_master/checkpoints/dota/traditional/traditional_resnet50_10.pt',
    #                    'batch_size': 4,
    #                    'log_wandb': 'off',
    #                    'emb_type': 'w2v'})
        
    #### Create the data loaders ####
    if parser.dataset == 'dota':
        train_setname = 'train2017seen'
        valseen_setname = 'test2017seen'
        valunseen_setname = 'train2017unseen'

    elif parser.dataset == 'pascalvoc':
        train_setname = 'train0712zsd'
        valseen_setname = 'testseen2007zsd'
        valunseen_setname = 'testunseen2007zsd'

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')
            
    if parser.coco_path is None:
        raise ValueError('Must provide --coco_path when training on COCO,')

    dataset_train = CocoDataset(parser.coco_path, set_name=train_setname, is_zsd=True,
                                transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    dataset_valseen = CocoDataset(parser.coco_path, set_name=valseen_setname, is_zsd=True,
                                transform=transforms.Compose([Normalizer(), Resizer()]))
    dataset_valunseen = CocoDataset(parser.coco_path, set_name=valunseen_setname, is_zsd=True,
                                transform=transforms.Compose([Normalizer(), Resizer()]))
    
    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)
    
    ### words embeddings
    class_embeddings = get_words(parser.dataset, parser.emb_type)
    vocab_embeddings = get_vocab(parser.dataset, parser.emb_type)
    # breakpoint()
    if parser.dataset == 'dota':
        s, u = 13, 3
    elif parser.dataset == 'pascalvoc':
        s, u = 16, 4

    #### Create the model #### 
    if parser.depth == 18:
        retinanet = model_polar.RetinaNet18(class_embeddings, vocab_embeddings, s+u, s, pretrained=True) 
    elif parser.depth == 34:
        retinanet = model_polar.RetinaNet34(class_embeddings, vocab_embeddings, s+u, s, pretrained=True)
    elif parser.depth == 50:
        retinanet = model_polar.RetinaNet50(class_embeddings, vocab_embeddings, s+u, s, pretrained=True)
    elif parser.depth == 101:
        retinanet = model_polar.RetinaNet101(class_embeddings, vocab_embeddings, s+u, s, pretrained=True)
    elif parser.depth == 152:
        retinanet = model_polar.RetinaNet152(class_embeddings, vocab_embeddings, s+u, s, pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()    

    # pretrained model dict
    pretrained_dict = torch.load(parser.traditional_model).state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items()}
    # load pretrained
    model_dict = retinanet.module.state_dict()
    for layer in pretrained_dict.keys():
        if layer in model_dict.keys():
            model_dict[layer] = pretrained_dict[layer]
    retinanet.module.load_state_dict(model_dict)
    print('Pretrained loaded!')

    print('Non trainable layers:')
    for k, v in retinanet.named_parameters():
        if v.requires_grad == False:
            print(k, v.requires_grad)

    # checkpoint 
    if parser.checkpoint_path is not None:
        checkpoint = torch.load(parser.checkpoint_path)
        retinanet.module.load_state_dict(checkpoint.state_dict())

    #### optimizers ####
    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    loss_hist = collections.deque(maxlen=500)
    
    loss_fn = PolarLoss()

    if parser.checkpoint_path is None:
        start_epoch = 0
        print('Start training...')
    else:
        start_epoch = int(parser.checkpoint_path.split('_')[-1].strip('.pt')) + 1
        print(f'Resume from epoch num {start_epoch}...')

    # breakpoint()

    #### train #### 
    epoch_num = 0

    # breakpoint()
    if ((epoch_num + 1) % 5) == 0 or (epoch_num == 0):
        ## evaluation traditional
        os.makedirs(f'log/{parser.dataset}', exist_ok=True)
        date = str(datetime.datetime.date(datetime.datetime.today()))
        print('Evaluating dataset')
        
        retinanet.eval()
        detect_type = 'traditional'
        stats_seen, class_aps_seen = coco_eval.evaluate_coco(parser.dataset, dataset_valseen, retinanet,emb_type = parser.emb_type, detect_type = detect_type, return_classap=True)

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

        f = open(f'log/{parser.dataset}/polar_mAPs.txt', 'a')
        f.writelines(lines)
        f.close()
        
        import pandas as pd 
        if parser.dataset == 'pascalvoc':
            unseen_classes = ['car', 'dog', 'sofa', 'train']
        elif parser.dataset == 'dota':
            unseen_classes = ['storage-tank', 'tennis-court', 'swimming-pool']
        if parser.dataset == 'coco':
            pass          
        seen_classes = [x for x in class_aps_seen.keys() if x not in unseen_classes]
        columns = ['Date', 'Dataset', 'SetName', 'DetectType', 'EmbType', 'Epoch', 'Alpha', 'Gamma', 'Beta'] + seen_classes + unseen_classes
        if os.path.exists(f'log/{parser.dataset}/polar_classAPs.csv'):
            f = pd.read_csv(f'log/{parser.dataset}/polar_classAPs.csv')
        else:
            f = pd.DataFrame(columns = columns, index=None)
        line = [date, parser.dataset,  valseen_setname, 'trad/zsd', parser.emb_type, epoch_num, loss_fn.alpha, loss_fn.gamma, loss_fn.beta]

        seen_aps = [np.round(v, 3) for (k, v) in class_aps_seen.items() if k in unseen_classes]
        unseen_aps = [np.round(v, 3) for (k, v) in class_aps_unseen.items() if k in unseen_classes]
        line += seen_aps
        line += unseen_aps
        f = pd.concat([f, pd.DataFrame([line], columns=columns)])
        f.to_csv(f'log/{parser.dataset}/polar_classAPs.csv', index = None)

import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt 
import seaborn as sns 

dataset = CocoDataset('../data/MSCOCO', set_name='val2014seenmini', is_zsd=True)

words = np.loadtxt('../data/MSCOCO/word_w2v.txt', delimiter = ',')
words = normalize(words, axis = 0).T

cos_sim = cosine_similarity(words, words)
f = plt.figure(figsize=(6,6))
sns.heatmap(cos_sim, cbar=True)
f.savefig('cosine_sim_author.png')

from retinanet.model_utils import get_words
words_me = get_words('coco')
cos_sim_me = cosine_similarity(words_me, words_me)
f = plt.figure(figsize=(6,6))
sns.heatmap(cos_sim_me, cbar=True)
f.savefig('cosine_sim_me.png')



for i in range(words.shape[0]):
    # if np.abs(words_me[i] - words[i]).sum() > 1:
        print(i, dataset.coco_labels[i], np.abs(words_me[i] - words[i]).sum())


from retinanet.model_utils import get_words
words_me = get_words('pascalvoc')
cos_sim_me = cosine_similarity(words_me, words_me)
f = plt.figure(figsize=(6,6))
sns.heatmap(cos_sim_me, cbar=True)
f.savefig('cosine_sim_me_voc.png')



vocab = np.loadtxt('../data/MSCOCO/vocabulary_w2v.txt', delimiter = ',')
vocab.shape 

from retinanet.model_utils import get_vocab
vocab_me = get_vocab('coco').T
vocab_me.shape 
i=2
vocab_me[:,i] - vocab[:,i]

flickr = open('../data/MSCOCO/FlickrTags.txt')
flickr = flickr.read()
flickr = flickr.split('\n')[:-1]
len(flickr) - 4738
if __name__ == '__main__':
    main()
