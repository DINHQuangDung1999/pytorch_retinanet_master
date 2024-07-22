import numpy as np
import time
import os
os.chdir('./pytorch_retinanet_master')
import shutil
import argparse
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
	UnNormalizer, Normalizer
from retinanet.detect_utils import detect_from_pred_boxes
from retinanet.model_utils import get_words, get_vocab
from tqdm import tqdm

def main(args=None):
	parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

	parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
	parser.add_argument('--coco_path', help='Path to COCO directory')
	parser.add_argument('--dump_dir', help='Path to save dir for detected images')
	parser.add_argument('--model_path', help='Path to model (.pt) file.')
	parser.add_argument('--detect_type', help='traditional, zsd or gzsd')
	parser.add_argument('--n_im', help='Number of images to detect.', default=50)
	parser.add_argument('--th', help='Threshold to plot.', default=0.5)
	parser = parser.parse_args(args)

	# from easydict import EasyDict
	# parser = EasyDict({'dataset': 'pascalvoc',
	# 					'coco_path': '../data/PascalVOC',
	# 					# 'dump_dir': '/home/qdinh/pytorch_retinanet_master/dump/pascalvoc/transductive/zsd',
	# 					# 'model_path': '/home/qdinh/pytorch_retinanet_master/checkpoints/pascalvoc/transductive/transductive_resnet50_0.pt',
	# 					# 'detect_type': 'zsd',
	# 					'dump_dir': '/home/qdinh/pytorch_retinanet_master/dump/pascalvoc/inductive/zsd',
	# 					'model_path': '/home/qdinh/pytorch_retinanet_master/checkpoints/pascalvoc/inductive/FL30/inductive_resnet50_29.pt',
	# 					'detect_type': 'zsd',
	# 					# 'dump_dir': '/home/qdinh/pytorch_retinanet_master/dump/pascalvoc/polar',
	# 					# 'model_path': '/home/qdinh/pytorch_retinanet_master/checkpoints/pascalvoc/polar/w2v/polar_resnet50_29.pt',
	# 					# 'detect_type': 'zsd',
	# 					# 'dump_dir': '/home/qdinh/pytorch_retinanet_master/dump/pascalvoc/traditional/zsd',
	# 					# 'model_path': '/home/qdinh/pytorch_retinanet_master/checkpoints/pascalvoc/traditional/traditional_resnet50_9.pt',
	# 					# 'detect_type': 'zsd',
	# 					# 'dump_dir': '/home/qdinh/pytorch_retinanet_master/dump/pascalvoc/vocab/zsd',
	# 					# 'model_path': '/home/qdinh/pytorch_retinanet_master/checkpoints/pascalvoc/polar/w2v/polar_resnet50_9.pt',
	# 					# 'detect_type': 'zsd',
	# 					'n_im': 200,
	# 					'th': 0.05})	
	 
	from easydict import EasyDict
	parser = EasyDict({'dataset': 'coco',
						'coco_path': '../data/MSCOCO',
						'dump_dir': '/home/qdinh/pytorch_retinanet_master/dump/coco/traditional',
						'model_path': '/home/qdinh/pytorch_retinanet_master/checkpoints/coco/traditional/traditional_resnet50_11.pt',
						'detect_type': 'traditional',
						'n_im': 100,
						'th': 0.05})	 
	
	if os.path.exists(parser.dump_dir):
		shutil.rmtree(parser.dump_dir)
	os.makedirs(parser.dump_dir,exist_ok=True)

	if parser.dataset == 'dota':
		if parser.detect_type == 'zsd':
			set_name = 'train2017unseen'
		elif parser.detect_type == 'gzsd':
			set_name = 'train2017mix'
		elif parser.detect_type == 'trad':
			set_name = 'val2017seen'
	elif parser.dataset == 'pascalvoc':
		if parser.detect_type == 'zsd':
			set_name = 'testunseen2007zsd'
		elif parser.detect_type == 'gzsd':
			set_name = 'testunseen2007zsd'
		else: 
			set_name = 'testseen2007zsd'
	elif parser.dataset == 'coco':
		if parser.detect_type == 'zsd':
			set_name = 'val2014unseen'
		elif parser.detect_type == 'gzsd':
			set_name = 'val2014mix'
		else: 
			set_name = 'val2014seenmini'
	else:
		raise ValueError('Dataset type not understood (must be csv or coco), exiting.')
	dataset_val = CocoDataset(parser.coco_path, set_name=set_name, is_zsd = True, transform=transforms.Compose([Normalizer(), Resizer()]))
	dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater)

	if torch.cuda.is_available():
		retinanet = torch.load(parser.model_path)
		retinanet = retinanet.cuda()
	else:
		retinanet = torch.load(parser.model_path, map_location=torch.device('cpu'))
	
	if parser.detect_type == 'zsd' or parser.detect_type == 'gzsd':
		class_embeddings = get_words(parser.dataset)		
		vocab_embeddings = get_vocab(parser.dataset)		
		if parser.dataset == 'dota':
			n_seen = 13
		elif parser.dataset == 'pascalvoc':
			n_seen = 16 
		elif parser.dataset == 'coco':
			n_seen = 65 
	else:
		class_embeddings = None 
		n_seen = None

	if torch.cuda.device_count() > 1:
		retinanet = torch.nn.DataParallel(retinanet).cuda()
	else:
		retinanet = torch.nn.DataParallel(retinanet)

	retinanet.eval()

	unnormalize = UnNormalizer()

	def draw_caption(image, box, caption, is_pred = False):
		if is_pred == False:
			b = np.array(box).astype(int)
			cv2.putText(image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
			cv2.putText(image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
		else:
			b = np.array(box).astype(int)
			cv2.putText(image, caption, (b[0], b[1] + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
			cv2.putText(image, caption, (b[0], b[1] + 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

	image_data = dataset_val.coco.dataset['images']
	image_names = [x['file_name'] for x in image_data]
	retinanet.module.classificationModel.emb_size = 300
	for idx, data in enumerate(tqdm(dataloader_val)):
		# break
		scale = data['scale'][0]
		with torch.no_grad():
			st = time.time()
			if torch.cuda.is_available():
				classifications, regressions, anchors = retinanet(data['img'].cuda().float())
			else:
				classifications, regressions, anchors = retinanet(data['img'].float())
			
			pred_scores, pred_labels, transformed_anchors = detect_from_pred_boxes(data['img'].cuda().float(), 
															classifications, regressions, anchors, 
															class_embeddings = class_embeddings, n_seen = n_seen, 
															detect_type = parser.detect_type, nboxes = 100)

			print('{} Elapsed time: {}'.format(image_names[idx], time.time()-st))

			idxs = np.where(pred_scores.cpu() > float(parser.th))

			img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

			img[img<0] = 0
			img[img>255] = 255

			img = np.transpose(img, (1, 2, 0))

			img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

			for j, pred_score in enumerate(pred_scores[idxs]):
				bbox = transformed_anchors[idxs[0][j], :]
				x1 = int(bbox[0])
				y1 = int(bbox[1])
				x2 = int(bbox[2])
				y2 = int(bbox[3])
				pred_label = int(pred_labels[idxs[0][j]])
				pred_id = dataset_val.coco_labels[pred_label]
				pred_name = dataset_val.labels[pred_id]
				caption = f'{pred_name} {np.round(pred_score.item(), 4)}'
				# draw pred box
				draw_caption(img, (x1, y1, x2, y2), caption, is_pred = True) 
				cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=1) 
				pass 
			gt_annot = dataset_val.load_annotations(idx)
			# breakpoint()
			for j in range(gt_annot.shape[0]):
				annot = gt_annot[j,:]
				x1 = int(annot[0]*scale) #multiply with scale
				y1 = int(annot[1]*scale)
				x2 = int(annot[2]*scale)
				y2 = int(annot[3]*scale)
				gt_label = int(annot[4])
				gt_id = dataset_val.coco_labels[gt_label]
				gt_name = dataset_val.labels[gt_id]
				# draw gt box
				draw_caption(img, (x1, y1, x2, y2), gt_name) 
				cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1) 
				pass
			cv2.imwrite(f'{parser.dump_dir}/{image_names[idx]}', img)
		if idx == int(parser.n_im):
			break

if __name__ == '__main__':
	main()