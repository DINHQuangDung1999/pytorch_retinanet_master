import json
from retinanet import model, model_inductive, model_transductive, model_vocab
import numpy as np

def get_words(dataset, type = 'w2v'):
    print(type, 'embeddings')
    if dataset == 'dota':
        
        f = open(f'../data/DOTA/{type}_words.json', 'r')
        class_embeddings = json.load(f)
        f = open(f'../data/DOTA/annotations/instances_test2017.json', 'r')
        f = json.load(f)
        class_map = {x['id']: x['name'] for x in f['categories']}

        seen_ids = [1,2,4,6,7,8,9,10,11,12,14,15,16]
        unseen_ids = [3,5,13]
 
    elif dataset == 'pascalvoc':
        if type == 'att':
            class_embeddings = np.loadtxt('../data/PascalVOC/VOC/VOC_att.txt', dtype=np.float32, delimiter = ',').T
            return class_embeddings
        else:
            f = open(f'../data/PascalVOC/{type}_words.json', 'r')
            class_embeddings = json.load(f)
        f = open(f'../data/PascalVOC/annotations/instances_test2007.json', 'r')
        f = json.load(f)
        class_map = {x['id']: x['name'] for x in f['categories']}

        seen_ids = [1,2,3,4,5,6,8,9,10,11,13,14,15,16,17,20]
        unseen_ids = [7,12,18,19]

    elif dataset == 'coco':

        f = open(f'../data/MSCOCO/{type}_words.json', 'r')
        class_embeddings = json.load(f)
        f = open(f'../data/MSCOCO/annotations/instances_val2014seenmini.json', 'r')
        f = json.load(f)
        class_map = {x['id']: x['name'] for x in f['categories']}        

        seen_ids = [1, 2, 3, 4, 6, 8, 9, 10, 11, 13, 15, 16, 18, 19, 20, 21, 22, 24, 25, 27, 28, 
                            31, 32, 35, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 55, 56, 
                            57, 59, 60, 61, 62, 63, 64, 65, 67, 72, 73, 75, 76, 77, 78, 79, 81, 82, 84, 85, 86, 87, 88, 90]
        unseen_ids = [5, 7, 14, 17, 23, 33, 34, 36, 48, 54, 58, 70, 74, 80, 89]

    tmp = []
    for i in seen_ids:
        tmp.append(class_embeddings[class_map[i]])
    for i in unseen_ids:
        tmp.append(class_embeddings[class_map[i]])
    class_embeddings = np.array(tmp)
    from sklearn.preprocessing import normalize
    class_embeddings = normalize(class_embeddings)
    
    return class_embeddings
def get_vocab(dataset, type = 'w2v'):
    print(type, 'embeddings')
    if dataset == 'dota':
        f = open(f'../data/DOTA/{type}_vocabulary.json', 'r')
    elif dataset == 'pascalvoc':
        f = open(f'../data/PascalVOC/{type}_vocabulary.json', 'r')
    elif dataset == 'coco':
        f = open(f'../data/MSCOCO/{type}_vocabulary.json', 'r')
    vocab_embeddings = json.load(f)
    tmp = []
    for i, key in enumerate(vocab_embeddings.keys()):
        tmp.append(vocab_embeddings[key])
    vocab_embeddings = np.array(tmp)
    from sklearn.preprocessing import normalize
    vocab_embeddings = normalize(vocab_embeddings)
    return vocab_embeddings
def get_model(dataset, depth, n_class_reg, n_class_cls, stage):
    if dataset == 'dota':
        s, u = 13, 3
    elif dataset == 'pascalvoc':
        s, u == 16, 4
    if stage == 'traditional':
        if depth == 18:
            retinanet_model =  model.RetinaNet18(n_class_reg, n_class_cls, pretrained=True) # train with only 13 seen classes
        elif depth == 34:
            retinanet_model =  model.RetinaNet34(n_class_reg, n_class_cls, pretrained=True)
        elif depth == 50:
            retinanet_model =  model.RetinaNet50(n_class_reg, n_class_cls, pretrained=True)
        elif depth == 101:
            retinanet_model =  model.RetinaNet101(n_class_reg, n_class_cls, pretrained=True)
        elif depth == 152:
            retinanet_model =  model.RetinaNet152(n_class_reg, n_class_cls, pretrained=True)
        else:
            raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    
    elif stage == 'inductive':
        if depth == 18:
            retinanet_model =  model_inductive.RetinaNet18(n_class_reg, n_class_cls, pretrained=True)
        elif depth == 34:
            retinanet_model =  model_inductive.RetinaNet34(n_class_reg, n_class_cls, pretrained=True)
        elif depth == 50:
            retinanet_model =  model_inductive.RetinaNet50(n_class_reg, n_class_cls, pretrained=True)
        elif depth == 101:
            retinanet_model =  model_inductive.RetinaNet101(n_class_reg, n_class_cls, pretrained=True)
        elif depth == 152:
            retinanet_model =  model_inductive.RetinaNet152(n_class_reg, n_class_cls, pretrained=True)
        else:
            raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    
    elif stage == 'transductive':
        if depth == 18:
            retinanet_model =  model_transductive.RetinaNet18(n_class_reg, n_class_cls, pretrained=True) 
        elif depth == 34:
            retinanet_model =  model_transductive.RetinaNet34(n_class_reg, n_class_cls, pretrained=True)
        elif depth == 50:
            retinanet_model =  model_transductive.RetinaNet50(n_class_reg, n_class_cls, pretrained=True)
        elif depth == 101:
            retinanet_model =  model_transductive.RetinaNet101(n_class_reg, n_class_cls, pretrained=True)
        elif depth == 152:
            retinanet_model =  model_transductive.RetinaNet152(n_class_reg, n_class_cls, pretrained=True)
        else:
            raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    
    return retinanet_model