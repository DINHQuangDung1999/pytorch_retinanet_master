import numpy as np
import os
os.chdir('./pytorch_retinanet_master')
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
	UnNormalizer, Normalizer
from tqdm import tqdm
import json 
from retinanet import model_transductive

def main(args=None):
	parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

	parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
	parser.add_argument('--coco_path', help='Path to COCO directory')
	parser.add_argument('--model_path', help='Path to model')
	parser.add_argument('--n_im', help='Number of images to visualize.', default=50)
	parser = parser.parse_args(args)

	from easydict import EasyDict
	parser = EasyDict({'dataset': 'pascalvoc',
						'coco_path': '../data/PascalVOC',
						'model_path': './checkpoints/pascalvoc/transductive/transductive_resnet50_2.pt',
						'n_im': 1})	
	

	if parser.dataset == 'dota':
		set_name = 'val2017zsd'
	elif parser.dataset == 'pascalvoc':
		set_name = 'test2007zsd'
	else:
		raise ValueError('Dataset type not understood (must be dota or pascalvoc), exiting.')

	dataset_gt = CocoDataset(parser.coco_path, set_name=set_name, is_zsd = True, transform=transforms.Compose([Normalizer(), Resizer()]))
	dataloader = DataLoader(dataset_gt, num_workers=1, collate_fn=collater)

	if parser.dataset == 'dota':
		set_name = 'val2017zsd'
		f = open('../data/dota_dataset/w2v_embeddings.json', 'r')
		w2v_embeddings = json.load(f)
		classes = list(w2v_embeddings.keys())
		seen_ids = [1,2,4,6,7,8,9,10,11,12,14,15,16]
		unseen_ids = [3,5,13]
		n_seen, n_unseen = 13, 3
	elif parser.dataset == 'pascalvoc':
		set_name = 'test2007zsd'
		f = open('../data/PascalVOC/w2v_embeddings.json', 'r')
		w2v_embeddings = json.load(f)
		classes = list(w2v_embeddings.keys())
		seen_ids = [1,2,3,4,5,6,8,9,10,11,13,14,15,16,17,20]
		unseen_ids = [7,12,18,19]		
		n_seen, n_unseen = 16, 4
	else:
		raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

	tmp = []
	for i in seen_ids:
		tmp.append(w2v_embeddings[classes[i-1]])
	for i in unseen_ids:
		tmp.append(w2v_embeddings[classes[i-1]])
	w2v_embeddings = np.array(tmp)
	from sklearn.preprocessing import normalize
	w2v_embeddings = normalize(w2v_embeddings)

	retinanet = model_transductive.RetinaNet50(n_seen + n_unseen, n_seen + n_unseen, True)
	# retinanet = RetinaNet50(n_seen + n_unseen, n_seen + n_unseen, True)
	if torch.cuda.is_available():
		retinanet.load_state_dict(torch.load(parser.model_path).state_dict())
		retinanet = torch.nn.DataParallel(retinanet).cuda()
	else:
		checkpoint = torch.load(parser.model_path, map_location=torch.device('cpu'))
		retinanet.load_state_dict(checkpoint.state_dict())

	image_data = dataset_gt.coco.dataset['images']
	image_names = [x['file_name'] for x in image_data]

	for idx, data in enumerate(tqdm(dataloader)):
		break 


	retinanet.train()
	classifications, regressions, anchors = retinanet(data['img'].cuda().float(), w2v_embeddings, n_seen, False)
	torch.round(classifications[0,100,:], decimals = 4)
	# Unseen predictions 
	with torch.no_grad():
		retinanet.eval()
		if torch.cuda.is_available():
			retinanet.eval()
			pred_scores, pred_labels, transformed_anchors = retinanet(data['img'].cuda().float(), w2v_embeddings, n_seen, False)
		else:
			pred_scores, pred_labels, transformed_anchors = retinanet(data['img'].float(), w2v_embeddings, n_seen)
		# break 
	breakpoint()


if __name__ == '__main__':
	main()