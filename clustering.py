import numpy as np
import math
import jieba
import random

import data_util
import simhash
import laplacian_eigenmap
import kmeans
import cnn_model

import torch

def getDiff(a, b):
    diff = 0
    a = a ^ b
    while a > 0:
        a = a & (a - 1)
        diff += 1
    return diff

def getMat(dic, corpus):
    ret = []
    for j in corpus:
        ls = []
        for word in j[0]:
            if dic.get(word, -1) != -1:
                added = dic[word]
            else:
                added = [0] * 300
            ls.append(added)
        ret.append(ls)
    return ret

def getFeature(dic, Corpus):
    corpus = Corpus[:1000] # remember to increse highly.
    sentences = [j[0] for j in corpus]
    hashedSentences = simhash.simhash(sentences, 32)
    B = hashedSentences

    # # you can try simhash directly, maybe it performs better.
    # hashedSentences = simhash.simhash(sentences, 128)
    # dataMat = np.array(hashedSentences)
    # lambda_, eigenVec_ = laplacian_eigenmap.laplacian_eigenmap(dataMat, 15, 32)

    input = getMat(dic, corpus)
    input = np.array(input)
    input = input.reshape(-1, 20 * 300)
    output = np.array(B)
    data = np.concatenate((input, output), axis=1)

    net = cnn_model.Net()
    criterion = cnn_model.nn.MSELoss()
    optimizer = cnn_model.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        generator = cnn_model.batch_generator(data, 100)
        i = 0
        for inputs, labels in generator:
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, h = net(torch.Tensor(inputs.reshape(-1, 1, 20, 300)))
            loss = criterion(outputs, torch.Tensor(labels.reshape(-1, 32)))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            i += 1
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    represented = []
    length = len(Corpus)
    for i in range(0, length, 500):
        input = getMat(dic, Corpus[i:min(length, i + 500)])
        input = np.array(input)
        input = input.reshape(-1, 1, 20, 300)
        outputs, h = net(torch.Tensor(input))
        represented += h.data.numpy().tolist()
        print('(%d %% %d) have been Embedding.' % (i, length))

    print('all sentences finished Embedding')

    return represented

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def uppickle(dict,file):
    import pickle
    with open(file, 'wb') as fo:
        pickle.dump(dict, fo)

if __name__ == '__main__':
    # dic = data_util.load_word2vec('./data/sgns.sogounews.bigram-char')
    # corpus = data_util.load_corpus('./data/toutiao_cat_data.txt')
    # 
    ## cut and supply the corpus
    # Corpus = corpus
    # for i, line in enumerate(Corpus):
    #     Corpus[i][0] = jieba.lcut(Corpus[i][0])
    #     length = len(Corpus[i][0])
    #     if length > 20:
    #         Corpus[i][0] = Corpus[i][0][:20]
    #     if length < 20:
    #         Corpus[i][0] += (20 - length) * ['']
    #     if i % 10000 == 0:
    #         print('(%d %% %d) sentences has been cutted' % (i, 382688))
    # uppickle(Corpus, './data/CuttedCorpus')

    # Corpus = unpickle('./data/CuttedCorpus')

    # random.seed(128)
    # random.shuffle(Corpus)

    # featureRepresent = getFeature(dic, Corpus)

    # uppickle(Corpus,'./data/Corpus')
    # uppickle(featureRepresent,'./data/featureRepresent')

    Corpus = unpickle('./data/Corpus')
    # featureRepresent = unpickle('./data/featureRepresent')

    # Sets, belongs = kmeans.kmeans([np.array(i) for i in featureRepresent], 15)

    # uppickle(belongs,'./data/belongs')

    belongs = unpickle('./data/belongs')

    random.seed(12)
    for i in range(len(belongs)):
#        belongs[i] = random.randint(0, 15 - 1)
        belongs[i] = i % 15

    sum = [0] * 15
    for i in belongs:
        sum[belongs[i]] += 1
    print('belongs are ', sum)

    label = [0] * 20
    for i in Corpus:
        label[i[1] - 101] += 1
    print('labels are ', label)

    random.seed()
    sum_score = 0
    total_score = 0


    maxrandlen = len(belongs)
    for x in range(0, maxrandlen):
        for y in range(0, maxrandlen):
            total_score += 1 if Corpus[x][1] == Corpus[y][1] else 1
            if Corpus[x][1] != Corpus[y][1]:
                sum_score += 1 if belongs[x] != belongs[y] else 0
            else:
                sum_score += 1 if belongs[x] == belongs[y] else 0
        if x % 1000 == 0:
            print('(%d %% %d) has been calced.' % (x, maxrandlen))

    print('final acc = %f %%' % (100 * sum_score / total_score)) 

    
