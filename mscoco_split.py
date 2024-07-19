import os 
import shutil 
import json 
import glob 
import pandas as pd 
import numpy as np 
os.chdir("./pytorch_retinanet_master")
from retinanet.dataloader import CocoDataset
from pycocotools.coco import COCO

df = pd.read_csv('/home/qdinh/data/MSCOCO/PLZSDannots/cls_names_seen_coco.csv', header = None)
seen_cats = list(df.loc[:,0])

df = pd.read_csv('/home/qdinh/data/MSCOCO/PLZSDannots/cls_names_test_coco.csv', header = None) 
all_cats = list(df.loc[:,0])
unseen_cats = list(set(all_cats) - set(seen_cats))

f = open('/home/qdinh/data/MSCOCO/annotations/instances_train2014.json')
f = json.load(f)

seen_cats_map = {}
for x in f['categories']:
    cat_name = x['name']
    if cat_name in seen_cats:
        id = x['id']
        seen_cats_map[cat_name] = id 
seen_cats_map.values()

unseen_cats_map = {}
for x in f['categories']:
    cat_name = x['name']
    if cat_name in unseen_cats:
        id = x['id']
        unseen_cats_map[cat_name] = id 
unseen_cats_map.values()

##### Train 
df = pd.read_csv('/home/qdinh/data/MSCOCO/PLZSDannots/train_coco_seen_all.csv', header = None)
img_names = set(df.iloc[:,0])
img_names = [os.path.basename(x) for x in img_names]
len(img_names)
len(df.loc[:,5].unique())

f = open('/home/qdinh/data/MSCOCO/annotations/instances_train2014.json')
f = json.load(f)

coco = COCO(f'/home/qdinh/data/MSCOCO/annotations/instances_train2014.json')
f = coco.dataset

f_annot = []
for annot in f['annotations']:
    id = annot['image_id']
    im_data = coco.loadImgs(id)[0]
    name = im_data['file_name']
    if name in img_names:
        f_annot.append(annot)

f_im = []
for im_data in f['images']:
    name = im_data['file_name']
    if im_data in f_im:
        continue
    if name in img_names:
        f_im.append(im_data)

f_cat = []
for i, category in enumerate(f['categories']):
    category['id'] = i + 1
    f_cat = category
new_f = f.copy()
new_f['annotations'] = f_annot
new_f['images'] = f_im

js = open('/home/qdinh/data/MSCOCO/annotations/instances_train2014seen.json', 'w')
json.dump(new_f, js)
js.close()

os.makedirs('/home/qdinh/data/MSCOCO/images/train2014seen', exist_ok=True)
for name in img_names:
    shutil.copy(f'/home/qdinh/data/MSCOCO/images/train2014/{name}',
                f'/home/qdinh/data/MSCOCO/images/train2014seen/{name}')


##### Val unseen
df = pd.read_csv('/home/qdinh/data/MSCOCO/PLZSDannots/validation_coco_unseen_all.csv', header = None)
img_names = set(df.iloc[:,0])
img_names = [os.path.basename(x) for x in img_names]
len(img_names)
print(len(df.loc[:,5]), len(df.loc[:,5].unique()))

coco = COCO(f'/home/qdinh/data/MSCOCO/annotations/instances_val2014.json')
f = coco.dataset

unseen_ids = set(unseen_cats_map.values())

f_annot = []
for annot in f['annotations']:
    id = annot['image_id']
    im_data = coco.loadImgs(id)[0]
    name = im_data['file_name']
    if name in img_names:
        category = annot['category_id']
        if category not in unseen_ids:
            continue
        f_annot.append(annot)

cat = []
for annot in f_annot:
    cat.append(annot['category_id'])
cat = set(cat)
cat 

f_im = []
for im_data in f['images']:
    name = im_data['file_name']
    if im_data in f_im:
        continue
    if name in img_names:
        f_im.append(im_data)

new_f = f.copy()
new_f['annotations'] = f_annot
new_f['images'] = f_im

js = open('/home/qdinh/data/MSCOCO/annotations/instances_val2014unseen.json', 'w')
json.dump(new_f, js)
js.close()

os.makedirs('/home/qdinh/data/MSCOCO/images/val2014unseen', exist_ok=True)
for name in img_names:
    shutil.copy(f'/home/qdinh/data/MSCOCO/images/val2014/{name}',
                f'/home/qdinh/data/MSCOCO/images/val2014unseen/{name}')







##### Val seen mini 
df = pd.read_csv('/home/qdinh/data/MSCOCO/PLZSDannots/validation_coco_seen_all.csv', header = None)
img_names = set(df.iloc[:,0])
img_names = [os.path.basename(x) for x in img_names]
len(img_names)
len(df.loc[:,5].unique())

coco = COCO(f'/home/qdinh/data/MSCOCO/annotations/instances_val2014.json')
f = coco.dataset

seen_ids = set(seen_cats_map.values())

count_seen = {}
for id in seen_ids:
    count_seen[id] = 0 

gt_im_ids = [x['id'] for x in f['images']]
import random
random.seed(10)
random.shuffle(gt_im_ids)
f_im = []
f_annot = []
for im_id in gt_im_ids:

    im_data = coco.loadImgs(im_id)[0]
    name = im_data['file_name']

    ann_ids = coco.getAnnIds(im_id) # get annots ids
    list_annots = coco.loadAnns(ann_ids) # list annots
    list_categories = [a['category_id'] for a in list_annots] # list categories

    if im_data in f_im:
        print('Existed')
        continue

    if name in img_names:
        f_im.append(im_data)
        f_annot += list_annots
        for cat_id in list_categories:
            if cat_id in count_seen.keys():
                count_seen[cat_id] += 1

    if min(count_seen.values()) > 50:
        break

# gt_im_ids.index(im_id)
len(f_im); len(f_annot)

# js = open('/home/qdinh/data/DOTA/annotations/instances_test2017seen.json', 'r')
# js = json.load(js)
# cat = []
# for annot in js['annotations']:
#     cat.append(annot['category_id'])
# np.unique(cat, return_counts = True)
new_f = f.copy()
new_f['annotations'] = f_annot
new_f['images'] = f_im

js = open('/home/qdinh/data/MSCOCO/annotations/instances_val2014seenmini.json', 'w')
json.dump(new_f, js)
js.close()

shutil.rmtree('/home/qdinh/data/MSCOCO/images/val2014seenmini')
os.makedirs('/home/qdinh/data/MSCOCO/images/val2014seenmini', exist_ok=True)
for im_data in f_im:
    name = im_data['file_name']
    shutil.copy(f'/home/qdinh/data/MSCOCO/images/val2014/{name}',
                f'/home/qdinh/data/MSCOCO/images/val2014seenmini/{name}')



##### Val seen
df = pd.read_csv('/home/qdinh/data/MSCOCO/PLZSDannots/validation_coco_seen_all.csv', header = None)
img_names = set(df.iloc[:,0])
img_names = [os.path.basename(x) for x in img_names]
len(img_names)
print(len(df.loc[:,5]), len(df.loc[:,5].unique()))

coco = COCO(f'/home/qdinh/data/MSCOCO/annotations/instances_val2014.json')
f = coco.dataset

unseen_ids = set(unseen_cats_map.values())

f_annot = []
for annot in f['annotations']:
    id = annot['image_id']
    im_data = coco.loadImgs(id)[0]
    name = im_data['file_name']
    if name in img_names:
        category = annot['category_id']
        if category not in unseen_ids:
            continue
        f_annot.append(annot)

cat = []
for annot in f_annot:
    cat.append(annot['category_id'])
cat = set(cat)
cat 

f_im = []
for im_data in f['images']:
    name = im_data['file_name']
    if im_data in f_im:
        continue
    if name in img_names:
        f_im.append(im_data)

new_f = f.copy()
new_f['annotations'] = f_annot
new_f['images'] = f_im

js = open('/home/qdinh/data/MSCOCO/annotations/instances_val2014seen.json', 'w')
json.dump(new_f, js)
js.close()

os.makedirs('/home/qdinh/data/MSCOCO/images/val2014seen', exist_ok=True)
for name in img_names:
    shutil.copy(f'/home/qdinh/data/MSCOCO/images/val2014/{name}',
                f'/home/qdinh/data/MSCOCO/images/val2014seen/{name}')
    




##### Val mix
df = pd.read_csv('/home/qdinh/data/MSCOCO/validation_coco_unseen_all.csv', header = None)
img_names = set(df.iloc[:,0])
img_names = [os.path.basename(x) for x in img_names]
len(img_names)
print(len(df.loc[:,5]), len(df.loc[:,5].unique()))

coco = COCO(f'/home/qdinh/data/MSCOCO/annotations/instances_val2014.json')
f = coco.dataset

unseen_ids = set(unseen_cats_map.values())

f_annot = []
for annot in f['annotations']:
    id = annot['image_id']
    im_data = coco.loadImgs(id)[0]
    name = im_data['file_name']
    if name in img_names:
        category = annot['category_id']
        if category not in unseen_ids:
            continue
        f_annot.append(annot)

cat = []
for annot in f_annot:
    cat.append(annot['category_id'])
cat = set(cat)
cat 

f_im = []
for im_data in f['images']:
    name = im_data['file_name']
    if im_data in f_im:
        continue
    if name in img_names:
        f_im.append(im_data)

new_f = f.copy()
new_f['annotations'] = f_annot
new_f['images'] = f_im

js = open('/home/qdinh/data/MSCOCO/annotations/instances_val2014unseen.json', 'w')
json.dump(new_f, js)
js.close()

os.makedirs('/home/qdinh/data/MSCOCO/images/val2014unseen', exist_ok=True)
for im_data in f['images']:
    name = im_data['file_name']
    if im_data in f_im:
        continue
    if name in img_names:
        shutil.copy(f'/home/qdinh/data/MSCOCO/images/val2014/{name}',
                    f'/home/qdinh/data/MSCOCO/images/val2014unseen/{name}')