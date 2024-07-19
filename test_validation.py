import os 
os.chdir('./pytorch_retinanet_master')
import datetime
import pandas as pd 
import numpy as np 
import argparse
import torch
from torchvision import transforms
from retinanet.dataloader import CocoDataset, Resizer, Normalizer
from retinanet.coco_eval import evaluate_coco
print('CUDA available: {}'.format(torch.cuda.is_available()))

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--model_path', help='Path to model', type=str) 
    
    parser = parser.parse_args(args)

    #### args ###
    from easydict import EasyDict
    parser = EasyDict({'dataset': 'pascalvoc',
                       'coco_path': '../data/PascalVOC',
                       'model_path': '/home/qdinh/pytorch_retinanet_master/checkpoints/pascalvoc/polar/polar_resnet50_29.pt',
                       })  
      
    #### Create the data loaders ####
    if parser.dataset == 'dota':
        valseen_setname = 'val2017seen'
        valunseen_setname = 'train2017unseen'

    elif parser.dataset == 'pascalvoc':
        valseen_setname = 'testseen2007zsd'
        valunseen_setname = 'testunseen2007zsd'

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    dataset_valseen = CocoDataset(parser.coco_path, set_name=valseen_setname, is_zsd=True,
                                transform=transforms.Compose([Normalizer(), Resizer()]))
    dataset_valunseen = CocoDataset(parser.coco_path, set_name=valunseen_setname, is_zsd=True,
                                transform=transforms.Compose([Normalizer(), Resizer()]))

    #### Create the model ####
    if torch.cuda.is_available():
        detector = torch.load(parser.model_path)
        detector = torch.nn.DataParallel(detector).cuda()
    else:
        detector = torch.load(parser.model_path, map_location=torch.device('cpu'))

    epoch_num = os.path.basename(parser.model_path).split('_')[-1].strip('.pt')
    detector.eval()
    detector.module.freeze_bn()

    os.makedirs(f'log/{parser.dataset}', exist_ok=True)
    date = str(datetime.datetime.date(datetime.datetime.today()))
    print('Evaluating dataset')
    
    detector.eval()

    detect_type = 'traditional'
    stats_seen, class_aps_seen = evaluate_coco(parser.dataset, dataset_valseen, detector, detect_type = detect_type, return_classap=True)
    lines= [
        f'Date: Date: {date} | Dataset: {parser.dataset} | Valset: {valseen_setname} | Epoch {epoch_num} | DetectType: {detect_type} \n'
        f'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {stats_seen[0]}\n'
        f'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {stats_seen[1]}\n'
        f'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {stats_seen[2]}\n'
        f'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {stats_seen[8]}\n'
    ]

    detect_type = 'zsd'
    stats_unseen, class_aps_unseen = evaluate_coco(parser.dataset, dataset_valunseen, detector, detect_type = detect_type, return_classap=True)
    lines += [
        f'Date: Date: {date} | Dataset: {parser.dataset} | Valset: {valunseen_setname} | Epoch {epoch_num} | DetectType: {detect_type} \n'
        f'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {stats_unseen[0]}\n'
        f'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {stats_unseen[1]}\n'
        f'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {stats_unseen[2]}\n'
        f'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {stats_unseen[8]}\n'
    ]
    # breakpoint()
    f = open(f'log/{parser.dataset}/test_mAPs.txt', 'a')
    f.writelines(lines)
    f.close()

    unseen_classes = ['car', 'dog', 'sofa', 'train']
    seen_classes = [x for x in class_aps_seen.keys() if x not in ['car', 'dog', 'sofa', 'train']]
    columns = ['Date', 'Dataset', 'SetName', 'DetectType', 'Epoch'] + seen_classes + unseen_classes
    f = pd.DataFrame(columns = columns, index=None)
    line = [date, parser.dataset,  valseen_setname, 'trad/zsd', epoch_num]

    seen_aps = [np.round(v, 3) for (k, v) in class_aps_seen.items() if k not in ['car', 'dog', 'sofa', 'train']]
    unseen_aps = [np.round(v, 3) for (k, v) in class_aps_unseen.items() if k in ['car', 'dog', 'sofa', 'train']]
    line += seen_aps
    line += unseen_aps
    f = pd.concat([f, pd.DataFrame([line], columns=columns)])
    f.to_csv(f'log/{parser.dataset}/test_classAPs.csv', index = None)

if __name__ == '__main__':
    main()
