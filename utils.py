import os
import torch
import numpy as np
import random
from itertools import dropwhile
import collections
import six
from allennlp.modules.elmo import batch_to_ids

def rindex(lst, item):
    def index_ne(x):
        return lst[x] != item
    try:
        return next(dropwhile(index_ne, reversed(range(len(lst)))))
    except StopIteration:
        raise ValueError("rindex(lst, item): item not in list")

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
    def __len__(self):
        return len(self.word2idx)

class SpaceTokenizer(object):
    def __init__(self):
        super(SpaceTokenizer, self).__init__()
    def tokenize(self, sent):
        return sent.split(' ')

class Corpus(object):
    def __init__(self, path, maxlen, lowercase=False, max_lines=-1, \
                 test_size=-1, train_path='train.txt', test_path='test.txt', tokenizer=None, \
                 label_dict=None):

        self.label_dict = label_dict
        self.maxlen = maxlen
        self.lowercase = lowercase
        self.train_path = os.path.join(path, train_path)
        self.test_path = os.path.join(path, test_path)
        self.max_lines = max_lines
        self.tokenizer = tokenizer if tokenizer else SpaceTokenizer()
        self.train = self.tokenize(self.train_path)
        if test_size > 0 and len(test_path) > 0:
            print("Test size and test path cannot both be present!")
            exit()
        if test_size > 0:
            print("Using {} in training set as test set".format(test_size))
            self.train, self.test = self.train[:-test_size], self.train[-test_size:]
            return
        elif len(test_path) > 0:
            print("Using {} as test set".format(test_path))
            self.test = self.tokenize(self.test_path)

    def tokenize(self, path):
        """Tokenizes a text file."""
        if self.label_dict:
            print("Convert class names in label_dict")

        cropped = 0.

        with open(path, 'r') as f:
            linecount = 0
            lines = []
            tags = []
            for line in f.readlines():
                linecount += 1
                if linecount % 10000 == 0: print("Read line", linecount, end='\r')
                if self.max_lines > 1 and linecount >= self.max_lines:
                    break
                if self.lowercase:
                    line = line.lower().strip().split('\t')
                else:
                    line = line.strip().split('\t')
                tag, sent = line[0], line[1]
                sent = self.tokenizer.tokenize(sent)
                if len(sent) > self.maxlen:
                    cropped += 1
                words = sent[:self.maxlen]
                if linecount == 2: print(words)
                lines.append(words)
                # Convert class label to int
                if self.label_dict:
                    tag = self.label_dict[tag]
                tags.append(tag)
        oov_count = -1
        # oov_count = print([(1 if ii==unk_idx else 0) for l in lines for ii in l])
        print("\nNumber of sentences cropped in {}: {:.0f} out of {:.0f} total, OOV {:.0f}".
              format(path, cropped, linecount, oov_count))

        return list(zip(tags, lines))


def batchify(data, bsz, shuffle=False, gpu=False):
    if shuffle:
        random.shuffle(data)
    tags, sents = zip(*data)
    nbatch = (len(sents)+bsz-1) // bsz
    # downsample biggest class
    # sents, tags = balance_tags(sents, tags)

    for i in range(nbatch):

        batch = sents[i*bsz:(i+1)*bsz]
        batch_tags = tags[i*bsz:(i+1)*bsz]
        # lengths = [len(x) for x in batch]
        # sort items by length (decreasing)
        # batch, batch_tags, lengths = length_sort(batch, batch_tags, lengths)

        # Pad batches to maximum sequence length in batch
        # find length to pad to

        # maxlen = lengths[0]
        # for b_i in range(len(batch)):
        #     pads = [pad_id] * (maxlen-len(batch[b_i]))
        #     batch[b_i] = batch[b_i] + pads
        # batch = torch.tensor(batch).long()
        batch = batch_to_ids(batch)
        batch_tags = torch.tensor(batch_tags).long()
        # lengths = [torch.tensor(l).long() for l in lengths]

        # yield (batch, batch_tags, lengths)
        yield (batch, batch_tags)

def filter_flip_polarity(data):
    flipped = []
    tags, sents = zip(*data)

    for i in range(len(tags)):
        org_tag = tags[i]
        sent = sents[i]
        if org_tag == 1: new_tag = 0
        if org_tag == 0: new_tag = 1
        flipped.append((new_tag, sent))
    print("Filtered and flipped {} sents from {} sents.".format(len(flipped), len(data)))
    return flipped

def length_sort(items, tags, lengths, descending=True):
    """In order to use pytorch variable length sequence package"""
    old_items = list(zip(items, tags, lengths))
    old_items.sort(key=lambda x: x[2], reverse=True)
    items, tags, lengths = zip(*old_items)
    return list(items), list(tags), list(lengths)

def balance_tags(items, tags):
    """Downsample largest group of tags"""
    new_items = []
    new_tags = []

    biggest_class = 2
    drop_ratio = .666
    for i in range(len(items)):
        tag = tags[i]
        item = items[i]
        if tag == biggest_class:
            if random.random() < drop_ratio:
                continue
        new_items.append(item)
        new_tags.append(tag)
    return new_items, new_tags

