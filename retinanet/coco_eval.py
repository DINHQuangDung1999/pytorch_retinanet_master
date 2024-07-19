from pycocotools.cocoeval import COCOeval
import json
import torch
import wandb
import numpy as np 
from tqdm import tqdm 
from retinanet.detect_utils import detect_from_pred_boxes
from retinanet.model_utils import get_vocab, get_words

def evaluate_coco(dataset_name, dataset, detector, emb_type = 'w2v', threshold=0.05, detect_type = 'zsd', loss_fn = None, log_wandb = None, return_classap = False):

    #### Create the model ####
    if dataset_name == 'dota':
        n_seen = 13
    elif dataset_name == 'pascalvoc':
        n_seen = 16
    elif dataset_name == 'coco':
        n_seen = 65

    if detect_type == 'zsd' or detect_type == 'gzsd':
        class_embeddings = get_words(dataset_name, type = emb_type)
    else:
        class_embeddings = None
        
    detector.eval()

    with torch.no_grad():

        # start collecting results
        results = []
        image_ids = []
        # epoch_loss = []
        for index in tqdm(range(len(dataset))):
            data = dataset[index]
            scale = data['scale']
            inputs = data['img'].permute(2, 0, 1).float().unsqueeze(dim=0)

            # run network
            if torch.cuda.is_available():
                classifications, regressions, anchors = detector(inputs.cuda())
                # annotations = data['annot'].cuda().unsqueeze(0)
            else:
                classifications, regressions, anchors = detector(inputs)
                # annotations = data['annot'].unsqueeze(0)
            
            scores, labels, boxes = detect_from_pred_boxes(inputs, 
                                                            classifications, regressions, anchors, 
                                                            class_embeddings, n_seen, detect_type, 
                                                            scorethresh= threshold, nboxes = None)
            # breakpoint()
            scores = scores.cpu()
            labels = labels.cpu()
            boxes  = boxes.cpu()

            # correct boxes for image scale
            boxes /= scale

            if boxes.shape[0] > 0:
                # change to (x, y, w, h) (MS COCO standard)
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]

                # compute predicted labels and scores
                #for box, score, label in zip(boxes[0], scores[0], labels[0]):
                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]

                    # scores are sorted, so we can break
                    if score < threshold:
                        break

                    # append detection for each positively labeled class
                    image_result = {
                        'image_id'    : dataset.image_ids[index],
                        'category_id': dataset.coco_labels[int(label)],
                        'score'       : float(score),
                        'bbox'        : box.tolist(),
                    }

                    # append detection to results
                    results.append(image_result)

            # append image to list of processed images
            image_ids.append(dataset.image_ids[index])

        if not len(results):
            print('No predictions!')
            return

        # write output
        json.dump(results, open('./bbox_results/{}_bbox_results.json'.format(dataset.set_name), 'w'), indent=4)

        # load results in COCO evaluation tool
        coco_true = dataset.coco
        coco_pred = coco_true.loadRes('./bbox_results/{}_bbox_results.json'.format(dataset.set_name))

        # run COCO evaluation
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        # breakpoint()
        # Extract class-wise average precision
        class_ap = {}
        for i, catId in enumerate(dataset.coco.getCatIds()):
            try:
                class_ap[dataset.coco.loadCats(catId)[0]['name']] = coco_eval.eval['precision'][0, :, i, 0, 2].mean()
            except:
                class_ap[dataset.coco.loadCats(catId)[0]['name']] = 0.

        detector.train()

        if return_classap == True:
            return coco_eval.stats, class_ap
        else:
            return coco_eval.stats

# # load results in COCO evaluation tool
# coco_true = dataset_valunseen.coco
# coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(dataset_valunseen.set_name))

# # run COCO evaluation
# coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
# # coco_eval.params.imgIds = image_ids
# coco_eval.evaluate()
# coco_eval.accumulate()
# coco_eval.summarize()
# # Extract class-wise average precision
# class_ap = {}
# for i, catId in enumerate(dataset_valunseen.coco.getCatIds()):
#     try:
#         class_ap[dataset_valunseen.coco.loadCats(catId)[0]['name']] = coco_eval.eval['precision'][0, :, i, 0, 2].mean()
#     except:
#         class_ap[dataset_valunseen.coco.loadCats(catId)[0]['name']] = 0.
 
# cat_ids = coco_true.getCatIds()
# cat_names = [cat['name'] for cat in coco_true.loadCats(cat_ids)]
# import itertools
# # Function to create a confusion matrix
# def create_confusion_matrix(coco_eval, cat_names):
#     num_classes = len(cat_names)
#     confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)

#     # Loop through each image
#     for img_id in coco_true.getImgIds():
#         # Get the ground truth and detection for the current image
#         gt_ann_ids = coco_true.getAnnIds(imgIds=[img_id], catIds=cat_ids)
#         gt_anns = coco_true.loadAnns(gt_ann_ids)
#         gt_categories = [ann['category_id'] for ann in gt_anns]
#         dt_ann_ids = coco_pred.getAnnIds(imgIds=[img_id], catIds=cat_ids)
#         dt_anns = coco_pred.loadAnns(dt_ann_ids)
#         dt_categories = [ann['category_id'] for ann in dt_anns]

#         # Create mappings from annotation IDs to categories
#         gt_category_mapping = {ann['id']: ann['category_id'] for ann in gt_anns}
#         dt_category_mapping = {ann['id']: ann['category_id'] for ann in dt_anns}
        
#         # Find matches using the coco_eval results
#         for iou_type, eval_imgs in enumerate(coco_eval.evalImgs[1]):

#             if eval_imgs is None:
#                 continue

#             for eval_img in eval_imgs:
#                 if eval_img is None or eval_img['image_id'] != img_id:
#                     continue

#                 gt_matches = eval_img['gtMatches']
#                 dt_matches = eval_img['dtMatches']

#                 for gt_id, dt_id in zip(gt_matches, dt_matches):
#                     if gt_id > 0 and dt_id > 0:
#                         gt_cat = gt_category_mapping[gt_id]
#                         dt_cat = dt_category_mapping[dt_id]
#                         confusion_matrix[cat_ids.index(gt_cat)][cat_ids.index(dt_cat)] += 1

#     return confusion_matrix

# confusion_matrix = create_confusion_matrix(coco_eval, cat_names)
# tmp = coco_eval.eval['precision']