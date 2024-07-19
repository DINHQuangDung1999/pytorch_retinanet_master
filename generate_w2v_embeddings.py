import numpy as np  
import gensim.downloader
import json
import argparse

def main(args=None):

    parser = argparse.ArgumentParser(description='Script for generating w2v embeddings (.txt)')

    parser.add_argument('--dir', help='Path to data directory', default = './data/DOTA')
    parser = parser.parse_args(args)

    model = gensim.downloader.load('word2vec-google-news-300')

    f = open('./data/DOTA/annotations/instances_val2017.json')
    f = json.load(f)
    category_names = []

    for category_data in f['categories']:
        category_names.append(category_data['name'])

    w2v_categories = {}
    for cat in category_names:
        try:
            compound_word = cat.split('-')
            emb = [model.get_vector(word) for word in compound_word]
            emb = list(np.mean(emb, axis = 0).astype(float))
            w2v_categories[cat] = emb
        except Exception as e:
            print(e)

    f = open('./data/DOTA/w2v_words.json', 'w')
    json.dump(w2v_categories, f)
    f.close()

    keys = list(w2v_categories.keys())
    f = open('./data/DOTA/FlickrTags.txt','r')
    lines = f.readlines()
    lines = [x[:-1] for x in lines]
    vocab = list(set(lines) - set(keys))
    w2v_vocab = {}
    for cat in vocab:
        try:
            compound_word = cat.split(' ')
            emb = [model.get_vector(word) for word in compound_word]
            emb = list(np.mean(emb, axis = 0).astype(float))
            w2v_vocab[cat] = emb
        except Exception as e:
            print(e)
    
    f = open('./data/DOTA/w2v_vocabulary.json', 'w')
    json.dump(w2v_vocab, f)
    f.close()

if __name__ == '__main__':
    main()