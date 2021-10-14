# -*- coding:utf8 -*-
"""
This py page is for the Modeling and training part of this project. 
Try to edit the place labeled "# TODO"!!!
"""

import torch
import torch.nn as nn
import torch.optim
import numpy as np


def word2index(word, vocab):
    """
    Convert an word token to an dictionary index
    """
    if word in vocab:
        value = vocab[word][0]
    else:
        value = -1
    return value


def index2word(index, vocab):
    """
    Convert an word index to a word token
    """
    for w, v in vocab.items():
        if v[0] == index:
            return w
    return 0


class Model(object):
    def __init__(self, args, vocab, pos_data, neg_data):
        """The Text Classification model """
        self.embeddings_dict = {}
        self.algo = args.algo
        if self.algo == "GLOVE":
            print("Now we are using the GloVe embedding")
            self.load_glove(args.emb_file)
        self.vocab = vocab
        self.pos_sentences = pos_data
        self.neg_sentences = neg_data
        self.lr = args.lr
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.dataset = []
        self.labels = []
        self.sentences = []

        self.train_data = []
        self.train_label = []

        self.valid_data = []
        self.valid_label = []

        """
        # TODO
        You should modify the code for the baseline Classifiers for self.algo == "GLOVE" (and otherwise use BOW) 
        shown below, it should have a high performance no less than 0.8 in terms of acc for the GLOVE condition. 
        You can replace or modify the classifier, but you must at least define the dimension for the output of 
        the linear layer (e.g., SIZE in nn.Linear(self.embed_size, SIZE), which needs to be the same as the 
        first arg of nn.Linear (don't change the 2, which corresponds to the number of classes).
        """
        if self.algo == "GLOVE":
            # TODO
            self.model = nn.Sequential(
                nn.Linear(self.embed_size, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 64),
                nn.Sigmoid(),
                nn.Linear(64, 2),
                nn.LogSoftmax(),
            )
        else:
            # TODO
            self.model = nn.Sequential(
                nn.Linear(len(self.vocab), 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 64),
                nn.Sigmoid(),
                nn.Linear(64, 2),
                nn.LogSoftmax(),
            )


    def load_dataset(self):
        """
        Load the training and testing dataset
        """
        for sentence in self.pos_sentences:
            new_sentence = []
            for l in sentence:
                if l in self.vocab:
                    if self.algo == "GLOVE":
                        new_sentence.append(l)
                    else:
                        new_sentence.append(word2index(l, self.vocab))
            self.dataset.append(self.sentence2vec(new_sentence, self.vocab))
            self.labels.append(0)
            self.sentences.append(sentence)

        for sentence in self.neg_sentences:
            new_sentence = []
            for l in sentence:
                if l in self.vocab:
                    if self.algo == "GLOVE":
                        new_sentence.append(l)
                    else:
                        new_sentence.append(word2index(l, self.vocab))
            self.dataset.append(self.sentence2vec(new_sentence, self.vocab))
            self.labels.append(1)
            self.sentences.append(sentence)

        indices = np.random.permutation(len(self.dataset))

        self.dataset = [self.dataset[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        self.sentences = [self.sentences[i] for i in indices]

        # split dataset
        test_size = len(self.dataset) // 10
        self.train_data = self.dataset[2 * test_size:]
        self.train_label = self.labels[2 * test_size:]

        self.valid_data = self.dataset[:2 * test_size]
        self.valid_label = self.labels[:2 * test_size]

    def rightness(self, predictions, labels):
        """ 
        Prediction of the error rate
        """
        pred = torch.max(predictions.data, 1)[1]
        rights = pred.eq(labels.data.view_as(pred)).sum()
        return rights, len(labels)

    def sentence2vec(self, sentence, dictionary):
        """ 
        Convert sentence text to vector representation 
        """
        """
        #TODO 
        You should modify the code to define two methods to convert sentence text to vector representations: 
        one is for Glove and another is for BOW. The first step is to set the size of the vectors, 
        which will be different for GLOVE and BOW. The next step is to create the vectors for your input sentences.
        Take a sentence vector to be the average of its word vectors.  Hint: Use numpy to init the vector; 
        Retrieve the GLOVE word vector from the embeddings_dict you create below, and retrieve the BOW vector 
        from self.vocab defined as part of the init for the class.
        """
        if self.algo == "GLOVE":

            #TODO
            sent_vec = np.tile(0, self.embed_size)
            # print(sent_vec)
            for each in sentence:
                if each in self.embeddings_dict:
                    # print(self.embeddings_dict[each])
                    # print(sent_vec)
                    sent_vec = np.add(sent_vec, self.embeddings_dict[each])
            sent_vec /= len(sentence)
            return sent_vec
        else:
            #TODO

            sent_vec = np.tile(0, len(dictionary))
            # print(len(dictionary))
            # print(sentence)
            for each in sentence:
                sent_vec[each] += 1
            # bow_index = sorted(dictionary.keys())
            # for each in sentence:
            #     if each in dictionary:
            #         sent_vec[bow_index.index(each)] += 1
            sent_vec = [x/len(sentence) for x in sent_vec]
            # print(np.sum(sent_vec))
            # exit()
            return sent_vec

    def load_glove(self, path):
        """
        Load Glove embeddings dictionary
        """
        """
        # TODO
        You should load the Glove embeddings from the local glove files like﻿"glove.8B.50d", 
        Then use "self.embeddings_dict" to store this words dict.
        """
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                # TODO
                self.embeddings_dict[word] = [float(x) for x in values[1:]]
        return 0

    def training(self):
        """
        The whole training and testing process.
        """
        losses = []
        """
        Note that the learning rate (lr) is a command line parameter.
        Here we provide a Cross entropy loss function 
        and an Adam optimizer, which includes the lr and model.parameters()
        If you choose, you can redefine the optimizer and loss_function
        """
        # TODO
        loss_function = torch.nn.NLLLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for epoch in range(10):
            print(epoch)
            for i, data in enumerate(zip(self.train_data, self.train_label)):
                x, y = data
                x = torch.tensor(x, requires_grad=True, dtype=torch.float).view(1, -1)
                y = torch.tensor(np.array([y]), dtype=torch.long)
                optimizer.zero_grad()
                # predict
                predict = self.model(x)
                # calculate loss
                loss = loss_function(predict, y)
                losses.append(loss.data.numpy())
                loss.backward()
                optimizer.step()
                # test every 1000 data
                if i % 1000 == 0:
                    val_losses = []
                    rights = []
                    for j, val in enumerate(zip(self.valid_data, self.valid_label)):
                        x, y = val
                        x = torch.tensor(x, requires_grad=True, dtype=torch.float).view(1, -1)
                        y = torch.tensor(np.array([y]), dtype=torch.long)
                        predict = self.model(x)
                        right = self.rightness(predict, y)
                        rights.append(right)
                        loss = loss_function(predict, y)
                        val_losses.append(loss.data.numpy())

                    right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
                    print('At the {} epoch，Training loss：{:.2f}, Testing loss：{:.2f}, Testing Acc: {:.2f}'.format(epoch, np.mean(losses),
                                                                                np.mean(val_losses), right_ratio))
        print("Training End")




