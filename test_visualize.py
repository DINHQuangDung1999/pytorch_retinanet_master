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
						'model_path': '/home/qdinh/pytorch_retinanet_master/checkpoints/pascalvoc/polar/polar_resnet50_10.pt',
						})  

	#### Create the model ####
	if torch.cuda.is_available():
		detector = torch.load(parser.model_path)
		detector = torch.nn.DataParallel(detector).cuda()
	else:
		detector = torch.load(parser.model_path, map_location=torch.device('cpu'))

	clas_emb = detector.module.classificationModel.class_embeddings
	voc_emb = detector.module.classificationModel.vocab_embeddings
	att = detector.module.classificationModel.wordvocabAttn
if __name__ == '__main__':
    main()