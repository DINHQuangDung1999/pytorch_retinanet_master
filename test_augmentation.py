import os 
os.chdir('./pytorch_retinanet_master')
from torchvision import transforms
from retinanet.dataloader import CocoDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
import albumentations as A 
import cv2 
import numpy as np 
import torch 
import pytorch_retinanet_master.retinanet.transforms as T

scales = [320, 336, 352, 368, 400, 416, 432, 448, 464, 480] #, 576, 608, 640]##
tr = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomSelect(
            T.RandomResize(scales, max_size=512),  #642
            T.Compose(
                [
                    T.RandomResize([400, 480, 512]), #500, 600
                    T.RandomSizeCrop(384, 512), #384, 600
                    T.RandomResize(scales, max_size=512), #642
                ]
            ),
        ),
        T.Normalize(),
        T.Resizer()
    ])


dataset= CocoDataset('../data/PascalVOC', set_name='train0712zsd', is_zsd=True, transform=tr)

example = dataset[0]
image = torch.from_numpy(example['img']).float().permute(2,0,1)
annot = example['annot']
annot = {'boxes': torch.from_numpy(annot[:,:4]).float(), 'labels': torch.from_numpy(annot[:,4]).long()}

# im, annot = normalize(image, annot)
# The image of ImageNet is relatively small.


augmented = tr(image, annot)

resizer = Resizer()
res = resizer(augmented[0], augmented[1])


image, target = augmented
annots = torch.concat([target['boxes'], target['labels'].unsqueeze(1)], dim = 1)
if 'scale' in target:
    annots['scale'] = target['scale']
augmented = {'img': image.permute(1,2,0), 'annot': annots}

# rr = RandomResize([400, 480, 512, 544], max_size=480)  #642
# augmented = rr(image, annot)
# image, target = augmented
# annots = torch.concat([target['boxes'], target['labels'].unsqueeze(1)], dim = 1)
# augmented = {'img': image.permute(1,2,0), 'annot': annots}

# rsc = RandomSizeCrop(224, 300)
# augmented = rsc(image, annot)
# image, target = augmented
# annots = torch.concat([target['boxes'], target['labels'].unsqueeze(1)], dim = 1)
# augmented = {'img': image.permute(1,2,0), 'annot': annots}

####################################################################
def draw_caption(image, box, caption, is_pred = False):
    if is_pred == False:
        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    else:
        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] + 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


# break
scale = 1
img = augmented['img'].numpy().copy()
# img = unnormalize(img)
img = np.array(255 * img)
img[img<0] = 0
img[img>255] = 255
img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
gt_annot = augmented['annot']
for j in range(gt_annot.shape[0]):
    annot = gt_annot[j,:]
    x1 = int(annot[0]*scale) #multiply with scale
    y1 = int(annot[1]*scale)
    x2 = int(annot[2]*scale)
    y2 = int(annot[3]*scale)
    gt_label = int(annot[4])
    gt_id = dataset.coco_labels[gt_label]
    gt_name = dataset.labels[gt_id]
    # draw gt box
    draw_caption(img, (x1, y1, x2, y2), gt_name) 
    cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1) 
    pass
cv2.imwrite(f'./figures/example.png', img)
