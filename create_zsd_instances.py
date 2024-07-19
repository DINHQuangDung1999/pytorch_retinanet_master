import shutil
import os 
os.chdir('./pytorch_retinanet_master')
import json 
from pycocotools.coco import COCO

# Unseen classes only
set_name = 'test2017'
dataset = 'MSCOCO'
type = 'seen'

coco = COCO(f'../data/{dataset}/annotations/instances_{set_name}.json')
f = coco.dataset
img_ids = [f['images'][i]['id'] for i in range(len(f['images']))]

def create_zsd_(dataset, set_name, type):

    if dataset == 'DOTA':
        seen_ids = [1,2,4,6,7,8,9,10,11,12,14,15,16]
        unseen_ids = [3,5,13] # tennis-court, swimming-pool, storage-tank
    elif dataset == 'PascalVOC':
        seen_ids = [1,2,3,4,5,6,8,9,10,11,13,14,15,16,17,20]
        unseen_ids = [7, 12, 18, 19] # car, dog, sofa, train     
    elif dataset == 'MSCOCO':    
        seen_ids = [1, 2, 3, 4, 6, 8, 9, 10, 11, 13, 15, 16, 18, 19, 20, 21, 22, 24, 25, 27, 28, 
                            31, 32, 35, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 55, 56, 
                            57, 59, 60, 61, 62, 63, 64, 65, 67, 72, 73, 75, 76, 77, 78, 79, 81, 82, 84, 85, 86, 87, 88, 90]
        unseen_ids = [5, 7, 14, 17, 23, 33, 34, 36, 48, 54, 58, 70, 74, 80, 89]

    train_img_ids = []
    train_annot_ids = []
    for j, im_id in enumerate(img_ids):
        ann_ids = coco.getAnnIds(im_id) # get annots ids
        list_annots = coco.loadAnns(ann_ids) # list annots
        ann_categories = [a['category_id'] for a in list_annots] # list categories

        # THE TRAIN SET CONTAIN ONLY INSTANCES FROM SEEN CLASSES
        seen_inter = set(ann_categories).intersection(seen_ids)
        unseen_inter = set(ann_categories).intersection(unseen_ids)
        if type == 'unseen':
            if len(seen_inter) == 0:
                train_img_ids.append(im_id)
                train_annot_ids += ann_ids
            else:
                continue
        if type == 'seen':
            if len(unseen_inter) == 0:
                train_img_ids.append(im_id)
                train_annot_ids += ann_ids
            else:
                continue 
        if type == 'mix':
            if len(seen_inter) > 0 and len(unseen_inter) > 0:
                train_img_ids.append(im_id)
                train_annot_ids += ann_ids
            else:
                continue       
        
    f_train_img = coco.loadImgs(train_img_ids)
    f_train_annot = coco.loadAnns(train_annot_ids)

    # Copy to a new folder
    if os.path.exists(f'../data/{dataset}/images/{set_name}{type}'):
        shutil.rmtree(f'../data/{dataset}/images/{set_name}{type}')
    os.makedirs(f'../data/{dataset}/images/{set_name}{type}', exist_ok= True)
    train_annot_im_name = [img_meta['file_name'] for img_meta in f_train_img]
    for im_name in train_annot_im_name:
        shutil.copy(f'../data/{dataset}/images/{set_name}/{im_name}', \
                    f'../data/{dataset}/images/{set_name}{type}/{im_name}')

    f_train = f.copy()
    f_train['annotations'] = f_train_annot 
    f_train['images'] = f_train_img

    js = open(f'../data/{dataset}/annotations/instances_{set_name}{type}.json', 'w')
    json.dump(f_train, js)
    js.close()

    print('Number of images/annotations: {}/{}'.format(len(f_train_img), len(f_train_annot)))
    cats = []
    for annot in f_train_annot:
        cats.append(annot['category_id'])
    cats = set(cats)
    print('Contained categories:', cats)

create_zsd_(dataset = dataset, set_name = set_name, type = 'seen')