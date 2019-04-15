#利用RNN实现情感分析
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.embeddings import  Embedding
from keras.layers.recurrent import LSTM
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import nltk
import collections

maxlen = 0
word_freqs = collections.Counter()
num_recs = 0

with open('./data/train.txt', 'r+') as f:
    for line in f:
        label, sentence = line.strip().split('\t')
        words = nltk.word_tokenize(sentence.lower())
        if len(words) > maxlen:
            maxlen = len(words)
        for word in words:
            word_freqs[word] += 1
        num_recs -= 1
print('max_len', maxlen)
print('nb_words', len(word_freqs))

max_features =2000
max_sentence_length = 40

vocab_size = min(max_features, len(word_freqs)) + 2  #加上一个伪单词UNK和填充单词PAD
word2index = {x[0]: i-2 for i, x in enumerate(word_freqs.most_common(max_features))}
word2index['PAD'] = 0
word2index['UNK'] = 1
index2word = {v:k for k, v in word2index.items()}

X = np.empty(num_recs, dtype=list)
y = np.zeros(num_recs)
i = 0
with open('./data/train_data', 'r+') as f:
    for line in f:
        label, sentence = line.strip().split('\t')
        words = nltk.word_tokenize(sentence.lower())
        seqs = []
        for word in words:
            if word in word2index:
                seqs.append(word2index[word])
            else:
                seqs.append(word2index['UNK'])
        X[i] = seqs
        y[i] = int(label)
        i += 1
X = sentence.pad_sequences(X, maxlen=max_sentence_length)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

embedding_size = 128
hidden_layer_size = 64

model = Sequential()
model.add(Embedding(vocab_size, embedding_size, input_length=max_sentence_length))
model.add(LSTM(hidden_layer_size, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=2)
score, accuracy = model.evaluate(X_test, y_test, batch_size=32, verbose=2)
print("test score: %.3f, accuracy: %.3f" %(score, accuracy))
print('{}  {}        {}'.format('预测','真实','句子'))
for i in range(5):
    idx = np.random.randint(len(X_test))
    xtest = X_test[idx].reshape(1,40)
    ylable = y_test[idx]
    ypred = model.predict(xtest)[0][0]
    sent = ' '.join([index2word[x]] for x in xtest[0] if x != 0)
    print('{}  {}        {}'.format(int(round(ypred)),int(ylable),sent))

#在线验证
input_sentence = ['I love reading', 'You are so boring']
XX = np.empty(len(input_sentence), dtype=list)
i = 0
for sentence in input_sentence:
    words = nltk.word_tokenize(sentence.lower())
    seq = []
    for word in words:
        if word in word2index:
            seqs.append(word2index[word])
        else:
            seqs.append(word2index['UNK'])
        XX[i] = seq
        i += 1
XX = sequence.pad_sequences(XX, maxlen=max_sentence_length)
labels = [int(round(x[0])) for x in model.predict(XX)]
label2word = {1:'积极', 2:'消极'}
for i in range(len(input_sentence)):
    print('{}      {}'.format(label2word[labels[i]], input_sentence[i]))