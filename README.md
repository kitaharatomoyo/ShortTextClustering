# ShortTextCLustering

mkdir ./data ,then put wordvec file and corpus file in ./data

### the format of wordvec file:
the first line: n m (the number of the word, and the length of each vector)
the other lines, each line's format is: word vec[0] vec[1] ... vec[m - 1] (the word itself, the vector of the word)

### the format of corpus:
each line: balabala_!_typeNumber_!_typeString_!_sentence_!_

### request package:
  jieba
  numpy
  pytorch 1.01

### important:
  first: if you don't want to change your corpus, you can change the file "data_util.py:'s function "load_corpus" to suit your corpus.
  second: you'd better use pytorch with gpu, here I just use cpu mode where you should change in file cnn_model.py . And in file clustering.py line 36, you'd better change the number 2000 bigger,like 10000 as you want to make the result better.
  third: in kmeans.py line 18, you can change the max_Iter=30 to a bigger number to get better result like 50, 100,but you will spend more time.
