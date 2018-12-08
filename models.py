import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os
import numpy as np
from allennlp.modules.elmo import Elmo

class SimpleELMOClassifier(nn.Module):
    def __init__(self, label_size, use_gpu, dropout=0.5):
        super(SimpleELMOClassifier, self).__init__()
        self.use_gpu = use_gpu
        self.dropout = dropout
        options_file = "elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        self.elmo = Elmo(options_file, weight_file, 1, dropout=dropout, do_layer_norm=False)
        # elmo output
#         Dict with keys:
#         ``'elmo_representations'``: ``List[torch.Tensor]``
#             A ``num_output_representations`` list of ELMo representations for the input sequence.
#             Each representation is shape ``(batch_size, timesteps, embedding_dim)``
#         ``'mask'``:  ``torch.Tensor``
#             Shape ``(batch_size, timesteps)`` long tensor with sequence mask.
        self.conv1 = nn.Conv1d(1024, 16, 3)
        self.p1 = nn.AdaptiveMaxPool1d(128)
        self.activation_func = nn.ReLU6()
        self.dropout_l = nn.Dropout(dropout)
        self.hidden2label = nn.Linear(2048, label_size)

    def init_weights(self):
        for name, param in self.hidden2label.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
        for name, param in self.conv1.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
        
    def forward(self, sentences):
        elmo_out = self.elmo(sentences)
        x = elmo_out['elmo_representations'][0]
        x = x.transpose(1,2)
        x = self.conv1(x)
        x = self.activation_func(x)
        x = self.p1(x)
        x = x.view(-1, 2048)
        x = self.dropout_l(x)
        y = self.hidden2label(x)
        return y

def load_models(load_path, model_args, suffix='', on_gpu=False):
    classifier = SimpleELMOClassifier(model_args['label_size'], \
                                      on_gpu, \
                                      model_args['dropout'])
    
    print('Loading models from', load_path)
    cls_path = os.path.join(load_path, "classifier_model{}.pt".format(suffix))
    classifier.load_state_dict(torch.load(cls_path))
    return classifier
