#!/usr/bin/env python

import argparse
import os
import time
import math
import numpy as np
import random
import sys
import json
import collections

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import Corpus, batchify
from models import SimpleELMOClassifier


parser = argparse.ArgumentParser(description='Text')
# Path Arguments
parser.add_argument('--data_path', type=str, required=True,
                    help='location of the corpus')
parser.add_argument('--out_dir', type=str, default='output',
                    help='output directory name')
parser.add_argument('--checkpoint', type=str, default='',
                    help='load checkpoint')

# Data Processing Arguments
parser.add_argument('--maxlen', type=int, default=128,
                    help='maximum sentence length')
parser.add_argument('--lowercase', action='store_true',
                    help='lowercase all text')

# Model Arguments
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')

# Training Arguments
parser.add_argument('--epochs', type=int, default=10,
                    help='maximum number of epochs')
parser.add_argument('--batch_size', type=int, default=200,
                    help='batch size')
parser.add_argument('--lr', type=float, default=2e-4,
                    help='learning rate')
parser.add_argument('--clip', type=float, default=5.,
                    help='gradient clipping, max norm')

# Evaluation Arguments
parser.add_argument('--log_interval', type=int, default=200,
                    help='interval to log training results')

# Other
parser.add_argument('--seed', type=int, default=1337,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use CUDA')

args = parser.parse_args()
print(vars(args))

def save_model(model, suffix=''):
    print("Saving model")
    with open('{}/model_{}.pt'.format(args.out_dir, suffix), 'wb') as f:
        torch.save(model.state_dict(), f)

###############################################################################
# Eval code
###############################################################################

def evaluate(model, data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    all_accuracies = 0.
    nbatches = 0.

    for batch in data:
        nbatches += 1.
        # source, tags, lengths = batch
        source, tags = batch
        if args.cuda:
            source = source.to("cuda")
            tags = tags.to("cuda")
        # output = model(source, lengths)
        output = model(source)
        max_vals, max_indices = torch.max(output, -1)

        accuracy = torch.mean(max_indices.eq(tags).float()).item()
        all_accuracies += accuracy
    return all_accuracies/nbatches

def train_classifier(args, classifier, train_batch, optimizer_, criterion_ce):
    classifier.train()
    classifier.zero_grad()
    # source, tags, lengths = train_batch
    source, tags = train_batch
    if args.cuda:
        source = source.to("cuda")
        tags = tags.to("cuda")

    # output: batch x nclasses
    # output = classifier(source, lengths)
    output = classifier(source)
    c_loss = criterion_ce(output, tags)

    c_loss.backward()

    # `clip_grad_norm` to prevent exploding gradient in RNNs / LSTMs
    torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.clip)
    optimizer_.step()

    total_loss = c_loss.item()

    # probs = F.softmax(output, dim=-1)
    # max_vals, max_indices = torch.max(probs, -1)
    # accuracy = torch.mean(max_indices.eq(tags).float()).item()

    return total_loss

def main(_):
    # make output directory if it doesn't already exist
    args.out_dir = os.path.join('.', args.out_dir)
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    # Set the random seed manually for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if args.no_cuda:
            print("WARNING: You have a CUDA device not used.")
        else:
            torch.cuda.manual_seed(args.seed)
    args.cuda = not args.no_cuda

    ###############################################################################
    # Load data
    ###############################################################################
    label_dict = dict([
                ('1', 0),
                ('2', 1),
                ])
    nclasses = len(label_dict)
    # create corpus
    corpus = Corpus(args.data_path,
                    maxlen=args.maxlen,
                    lowercase=args.lowercase,
                    max_lines=-1,
                    test_size=0,
                    train_path='train.txt',
                    test_path='test.txt',
                    # tokenizer=tokenizer,
                    label_dict=label_dict,
                    )
    args.nclasses = nclasses
    # save arguments
    with open('{}/args.json'.format(args.out_dir), 'w') as f:
        json.dump(vars(args), f)

    eval_batch_size = 20

    # Print corpus stats
    class_counts = collections.Counter([c[0] for c in corpus.train])
    print("Train: {}".format(class_counts))
    class_counts = collections.Counter([c[0] for c in corpus.test])
    print("Test: {}".format(class_counts))

    train_data = batchify(corpus.train, args.batch_size, shuffle=True)
    test_data = batchify(corpus.test, eval_batch_size, shuffle=False)

    print("Loaded data!")

    ###############################################################################
    # Build the models
    ###############################################################################

    classifier = SimpleELMOClassifier(label_size=args.nclasses, use_gpu=args.cuda, dropout=args.dropout,)
    # print(classifier)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Num params:", count_parameters(classifier))
    # optimizer = optim.Adam(classifier.parameters(), lr=args.lr)
    # optimizer = optim.RMSprop(classifier.parameters(), lr=args.lr)
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=1e-4)
    learning_rate_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    criterion_ce = nn.CrossEntropyLoss()

    if args.cuda:
        classifier.to("cuda")
        criterion_ce = criterion_ce.to("cuda")

    classifier.init_weights()
    if args.checkpoint:
        classifier.load_state_dict(torch.load(args.checkpoint))

    print("Training...")
    with open("{}/logs.txt".format(args.out_dir), 'w') as f:
        f.write('Training...\n')
    niter_global = 0
    for epoch in range(1, args.epochs+1):
        print("Epoch ", epoch)

        # loop through all batches in training data
        for train_batch in train_data:
            loss = train_classifier(args, classifier, train_batch, optimizer, criterion_ce)
            niter_global += 1
            if niter_global % 10 == 0:
                msg = 'loss {:.5f}'.format(loss)
                print(msg, end='\r')
                with open("{}/logs.txt".format(args.out_dir), 'a') as f:
                    f.write(msg)
                    f.write('\n')
                    f.flush()

            if niter_global % 1000 == 0:
                with torch.no_grad():
                    accuracy = evaluate(classifier, test_data)
                msg = 'test acc {:.4f}'.format(accuracy)
                # msg = 'test loss {:.5f} acc {:.2f}'.format(test_loss, accuracy)
                print('\n' + msg)
                with open("{}/logs.txt".format(args.out_dir), 'a') as f:
                    f.write(msg)
                    f.write('\n')
                    f.flush()
                # save_model(classifier, suffix=niter_global)
                # print("Saved model step {}".format(niter_global))
                # we use generator, so must re-gen test data
                test_data = batchify(corpus.test, eval_batch_size, shuffle=False)

        # end of epoch ----------------------------
        # save model every epoch
        save_model(classifier, suffix=epoch)
        print("saved model epoch {}".format(epoch))

        # clear cache between epoch
        torch.cuda.empty_cache()
        # decay learning rate
        learning_rate_scheduler.step()
        # shuffle between epochs
        train_data = batchify(corpus.train, args.batch_size, shuffle=True)


if __name__ == "__main__":
    main(1)
