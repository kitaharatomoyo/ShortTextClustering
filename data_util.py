import math
import numpy as np

def load_word2vec(file_path):
    dic = {}
    with open(file_path, 'r') as f:
        n, m = f.readline().split()
        i = 0
        for line in f:
            i += 1
            ls = line.split()
            word = ls[0]
            vec = [float(j) for j in ls[-300:]]
            dic[word] = vec
            if i % 50000 == 0:
                print('(%d %% %s) words\' vector loaded;' % (i, n))
    print('load_dictionary finished.')
    return dic

def load_corpus(file_path):
    corpus = []
    with open(file_path, 'r', encoding='utf-8') as f:
        i = 0
        for line in f:
            i += 1
            sentence = line[:-1]
            tagNumber = 0
            tagName = 'byouki'
            corpus.append([sentence, tagNumber, tagName])
            if i % 50000 == 0:
                print('(%d %% %s) sentences loaded;' % (i, 382688))
    return corpus

if __name__ == '__main__':
    # dic = load_word2vec('./data/sgns.sogounews.bigram-char')
    # print(len(dic))
    corpus = load_corpus('./data/toutiao_cat_data.txt')
    print(len(corpus))
    