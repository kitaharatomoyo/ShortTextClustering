import numpy as np
import math
import jieba
import random
import time
import torch

import data_util
import simhash
import laplacian_eigenmap
import kmeans
import cnn_model

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
    corpus = Corpus[:5000] # remember to increse highly.
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

    print('Start Traning Cnn')

    i = 0
    running_loss = 0.0
    for epoch in range(50):  # loop over the dataset multiple times

        generator = cnn_model.batch_generator(data, 100)
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
            if i % 20 == 19:    # print every 100 mini-batches
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

def consist(lt):
    ret = ''
    for i in lt:
        ret += i
    return ret

if __name__ == '__main__':

    start_time = time.process_time()

    dic = data_util.load_word2vec('./data/sgns.sogounews.bigram-char')
    corpus = data_util.load_corpus('./data/ganmao.txt')
    
    # cut and supply the corpus
    Corpus = corpus
    CorpusLength = len(Corpus)
    for i, line in enumerate(Corpus):
        Corpus[i][0] = jieba.lcut(Corpus[i][0])
        length = len(Corpus[i][0])
        if length > 20:
            Corpus[i][0] = Corpus[i][0][:20]
        if length < 20:
            Corpus[i][0] += (20 - length) * ['']
        if i % 10000 == 0:
            print('(%d %% %d) sentences has been cutted' % (i, CorpusLength))
    uppickle(Corpus, './data/CuttedCorpus')

    Corpus = unpickle('./data/CuttedCorpus')

    random.seed()
    random.shuffle(Corpus)

    featureRepresent = getFeature(dic, Corpus)

    uppickle(Corpus,'./data/Corpus')
    uppickle(featureRepresent,'./data/featureRepresent')

    Corpus = unpickle('./data/Corpus')
    featureRepresent = unpickle('./data/featureRepresent')

    print('data has been loaded.\n kmeans start')
    Sets, belongs = kmeans.kmeans([np.array(i) for i in featureRepresent], 300)

    uppickle(belongs,'./data/belongs')

    belongs = unpickle('./data/belongs')

    label_sentence = [[] for i in range(300)]

    for i, belong in enumerate(belongs):
        label_sentence[belong].append(consist(Corpus[i][0]))

    with open('ganmao_clustering.txt', 'w') as w:
        for i in range(300):
            w.write(str(i) + '\n')
            w.write(str(len(label_sentence[i])) + '\n')
            for sentence in label_sentence[i]:
                w.write(sentence + '\n')
