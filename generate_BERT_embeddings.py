import os 
os.chdir('./pytorch_retinanet_master')
import random 
import torch
from transformers import BertTokenizer, BertModel
from retinanet.dataloader import CocoDataset
import json 

def get_embedding(cat):
    marked_text = "[CLS] " + cat + " [SEP]"
    tokenized = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized)
    segments_ids = [1] * len(tokenized)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    with torch.no_grad():
        encoding = model(tokens_tensor, segments_tensors)[2]
    encoding = torch.stack(encoding, dim = 0)
    encoding = torch.squeeze(encoding, dim=1)
    encoding = encoding.permute(1,0,2)
    
    token_vecs_sum = []
    for token in encoding:
        sum_vec = torch.sum(token[-4:], dim=0) # Sum the vectors from the last four layers.
        token_vecs_sum.append(sum_vec)
    encoding = torch.mean(torch.stack(token_vecs_sum, dim = 0), dim =0)
    encoding = encoding/torch.sqrt(torch.sum(encoding**2))
    encoding = list(encoding.cpu().numpy().astype(float))

    return encoding 

dataset = CocoDataset('../data/PascalVOC', set_name= 'test2007', is_zsd=True)
categories = list(dataset.classes.keys())

# Set a random seed
random_seed = 42
random.seed(random_seed)
 
# Set a random seed for PyTorch (for GPU as well)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
    
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
model.eval()

bert_embeddings = {}
for cat in categories:
    encoding = get_embedding(cat)
    bert_embeddings[cat] = encoding

f = open('../data/PascalVOC/bert_words.json', 'w')
json.dump(bert_embeddings, f)
f.close()

keys = list(bert_embeddings.keys())
f = open('../data/PascalVOC/FlickrTags.txt','r')
lines = f.readlines()
lines = [x[:-1] for x in lines]
vocab = list(set(lines) - set(keys))

bert_vocab = {}
for cat in vocab:
    encoding = get_embedding(cat)
    bert_vocab[cat] = encoding

f = open('../data/PascalVOC/bert_vocabulary.json', 'w')
json.dump(bert_vocab, f)
f.close()

# set(bert_vocab.keys()).intersection(set(bert_embeddings.keys()))