import os 
os.chdir('./pytorch_retinanet_master')

from pycocotools.cocoeval import COCOeval
import json
from tqdm import tqdm 
import numpy as np 
import argparse
import torch
from torchvision import transforms

from retinanet.dataloader import CocoDataset, Resizer, Normalizer
from retinanet.coco_eval import evaluate_coco
# assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--set_name', help='validation set name', default = 'val2017')
    parser.add_argument('--detect_type', help='traditional, zsd or gzsd', default = 'zsd')
    parser.add_argument('--stage', help='traditional, inductive, transductive or polar', type=str) 
    parser.add_argument('--model_path', help='Path to model', type=str) 
    
    parser = parser.parse_args(args)

    #### args Pascal ###
    from easydict import EasyDict
    parser = EasyDict({'dataset': 'pascalvoc',
                       'coco_path': '../data/PascalVOC',
                       'set_name': 'testunseen2007zsd',
                       'model_path': '/home/qdinh/pytorch_retinanet_master/checkpoints/pascalvoc/inductive/FL30/inductive_resnet50_9.pt',
                       'detect_type': 'zsd'
                       })  

    
    # #### args COCO ###
    # from easydict import EasyDict
    # parser = EasyDict({'dataset': 'coco',
    #                    'coco_path': '../data/MSCOCO',
    #                    'set_name': 'val2014seen',
    #                    'model_path': '/home/qdinh/pytorch_retinanet_master/checkpoints/coco/traditional/traditional_resnet50_7.pt',
    #                    'detect_type': 'trad'
    #                    })  

    # #### args DOTA ###
    # from easydict import EasyDict
    # parser = EasyDict({'dataset': 'dota',
    #                    'coco_path': '../data/DOTA',
    #                    'set_name': 'train2017unseen',
    #                    'model_path': '/home/qdinh/pytorch_retinanet_master/checkpoints/dota/traditional/traditional_resnet50_9.pt',
    #                    'detect_type': 'zsd'
    #                    })  

    print(parser.model_path)
    dataset_val = CocoDataset(parser.coco_path, set_name=parser.set_name, is_zsd=True,
                              transform=transforms.Compose([Normalizer(), Resizer()]))

    #### Create the model ####
    if torch.cuda.is_available():
        detector = torch.load(parser.model_path).cuda()
        detector = torch.nn.DataParallel(detector)
    else:
        detector = torch.load(parser.model_path, map_location=torch.device('cpu'))

    detector.eval()
    detector.module.freeze_bn()
    detector.module.classificationModel.emb_size = 300
    # breakpoint()
    stats, classap = evaluate_coco(parser.dataset, dataset_val, detector, detect_type = parser.detect_type, return_classap=True)
    print(classap)

if __name__ == '__main__':
    main()
