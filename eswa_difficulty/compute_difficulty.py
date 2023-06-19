import os
from random import randrange
from statistics import mean

import eswa_difficulty.utils
from eswa_difficulty.piano_fingering.compute_embeddings import compute_embedding_score
from eswa_difficulty.utils import prediction2label
import torch
from torch import nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as packer, pad_packed_sequence as padder, PackedSequence
import torch.nn.functional as F
import torch


class ordinal_loss(nn.Module):
    """Ordinal regression with encoding as in https://arxiv.org/pdf/0704.1028.pdf"""
    def __init__(self, weight_class=False):
        super(ordinal_loss, self).__init__()
        self.weights = weight_class

    def forward(self, predictions, targets):
        # Fill in ordinalCoefficientVariationLoss target function, i.e. 0 -> [1,0,0,...]
        modified_target = torch.zeros_like(predictions)
        for i, target in enumerate(targets):
            modified_target[i, 0:target+1] = 1
        # loss
        if self.weights is not None:
            return torch.sum((self.weights * F.mse_loss(predictions, modified_target, reduction="none")).mean(axis=1))
        else:
            return torch.sum((F.mse_loss(predictions, modified_target, reduction="none")).mean(axis=1))


class torch_tuple:
    def __init__(self, rh, lh, exp=None):
        self.rh = rh
        self.lh = lh
        self.exp = exp

    def to(self, dev):
        self.rh = self.rh.to(dev)
        self.lh = self.lh.to(dev)
        if self.exp is not None:
            self.exp = self.exp.to(dev)
        return self

    def short_end(self):
        self.rh = self.rh if self.rh.shape[0] < 400 else self.rh[-400:, :, :]
        self.lh = self.lh if self.lh.shape[0] < 400 else self.lh[-400:, :, :]
        if self.exp is not None:
            self.exp = self.exp if self.exp.shape[0] < 400 else self.exp[-400:, :, :]
        return self

    def unsqueeze(self, dim):
        self.rh = self.rh.unsqueeze(dim)
        self.lh = self.lh.unsqueeze(dim)
        return self

    def _random_excerpt(self, mat):
        len_mat = mat.shape[0]
        idx = randrange(0, len_mat - 400)
        return mat[idx:idx+400, :, :]

    def short_random(self):
        self.rh = self.rh if self.rh.shape[0] <= 400 else self._random_excerpt(self.rh)
        self.lh = self.lh if self.lh.shape[0] <= 400 else self._random_excerpt(self.lh)
        if self.exp is not None:
            self.exp = self.exp if self.exp.shape[0] <= 400 else self._random_excerpt(self.exp)
        return self

    def get_rh(self):
        return self.rh

    def float(self):
        return self

    def get_lh(self):
        return self.lh

    def concat(self, new_tuple):
        self.rh = torch.concat([self.rh, new_tuple.rh], dim=2)
        self.lh = torch.concat([self.lh, new_tuple.lh], dim=2)
        return self

    def get_exp(self):
        return self.exp



class ContextAttention(nn.Module):
    def __init__(self, size, num_head):
        super(ContextAttention, self).__init__()
        self.attention_net = nn.Linear(size, size)
        self.num_head = num_head

        if size % num_head != 0:
            raise ValueError("size must be dividable by num_head", size, num_head)
        self.head_size = int(size / num_head)
        self.context_vector = torch.nn.Parameter(torch.Tensor(num_head, self.head_size, 1))
        nn.init.uniform_(self.context_vector, a=-1, b=1)

    def get_attention(self, x):
        attention = self.attention_net(x)
        attention_tanh = torch.tanh(attention)
        # attention_split = torch.cat(attention_tanh.split(split_size=self.head_size, dim=2), dim=0)
        attention_split = torch.stack(attention_tanh.split(split_size=self.head_size, dim=2), dim=0)
        similarity = torch.bmm(attention_split.view(self.num_head, -1, self.head_size), self.context_vector)
        similarity = similarity.view(self.num_head, x.shape[0], -1).permute(1, 2, 0)
        return similarity

    def forward(self, x):
        attention = self.attention_net(x)
        attention_tanh = torch.tanh(attention)
        if self.head_size != 1:
            attention_split = torch.stack(attention_tanh.split(split_size=self.head_size, dim=2), dim=0)
            similarity = torch.bmm(attention_split.view(self.num_head, -1, self.head_size), self.context_vector)
            similarity = similarity.view(self.num_head, x.shape[0], -1).permute(1, 2, 0)
            similarity[x.sum(-1) == 0] = -1e10  # mask out zero padded_ones
            softmax_weight = torch.softmax(similarity, dim=1)

            x_split = torch.stack(x.split(split_size=self.head_size, dim=2), dim=2)
            weighted_x = x_split * softmax_weight.unsqueeze(-1).repeat(1, 1, 1, x_split.shape[-1])
            attention = weighted_x.view(x_split.shape[0], x_split.shape[1], x.shape[-1])
        else:
            softmax_weight = torch.softmax(attention, dim=1)
            attention = softmax_weight * x

        sum_attention = torch.sum(attention, dim=1)
        return sum_attention


class gru_squared(nn.Module):
    def __init__(self, input_size, dropout, num_layers, hidden_size, batch_first, bidirectional):
        super(gru_squared, self).__init__()
        self.gru1 = nn.GRU(
            input_size=input_size,
            num_layers=num_layers,
            dropout=dropout,
            hidden_size=hidden_size,
            batch_first=batch_first,
            bidirectional=bidirectional
        )

    def forward(self, x, x_lengths):
        x_packed = packer(x.float(), x_lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        output, _ = self.gru1(x_packed)
        output_padded, _ = padder(output, batch_first=True)
        return output_padded


class dropout_gru_reduced_relu(nn.Module):
    def __init__(self, input_size=10, hidden_size=16, num_layers=3):
        super(dropout_gru_reduced_relu, self).__init__()
        self.dropout = nn.Dropout(p=0.2)

        self.gru1 = gru_squared(
            input_size=input_size,
            num_layers=num_layers,
            dropout=0.2,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x, x_lengths):
        x = self.dropout(x)
        output = self.gru1(x, x_lengths)
        return output


class PerformanceSummariser_ordinal(nn.Module):
    def __init__(self, n_classes=3, input_size=10):
        super(PerformanceSummariser_ordinal, self).__init__()
        #  input is (batchSize x time x 64)
        size = 64
        self.branch_rh = dropout_gru_reduced_relu(input_size=input_size, hidden_size=size//2, num_layers=3)

        self.attention_rh = ContextAttention(size=size, num_head=1)
        self.non_linearity = nn.ReLU()
        self.FC = nn.Linear(size, n_classes)

    def forward(self, fng_rh, rh_length):
        fng_rh = self.branch_rh(fng_rh, rh_length)
        summary_rh = self.attention_rh(fng_rh)
        classification = self.non_linearity(summary_rh)
        classification = self.FC(classification)
        return classification


class PerformanceSummariser_argnn_ordinal(nn.Module):
    def __init__(self, input_size=64, number_classes=3):
        super(PerformanceSummariser_argnn_ordinal, self).__init__()
        hidden_size = 64
        self.branch_lh = dropout_gru_reduced_relu(input_size=input_size, hidden_size=hidden_size, num_layers=3)
        self.branch_rh = dropout_gru_reduced_relu(input_size=input_size, hidden_size=hidden_size, num_layers=3)
        size = hidden_size*2
        self.attention_rh = ContextAttention(size=size, num_head=1)
        self.attention_lh = ContextAttention(size=size, num_head=1)
        self.non_linearity = nn.ReLU()
        self.FC = nn.Linear(size*2, number_classes)

    def forward(self, feat, length):
        fng_rh = self.branch_rh(feat.get_rh(), length.get_rh())
        summary_rh = self.attention_rh(fng_rh)
        fng_lh = self.branch_lh(feat.get_lh(), length.get_lh())
        summary_lh = self.attention_lh(fng_lh)
        classification = self.non_linearity(torch.concat([summary_rh, summary_lh], dim=1))
        classification = self.FC(classification)
        return classification

def get_pred(feats, device, models, rep):
    y_pred = []
    feats = feats.unsqueeze(0)
    if rep == "argnn":
        length_rh = feats.rh.shape[0]
        length_lh = feats.lh.shape[1]
        length = torch_tuple(torch.Tensor([length_rh]), torch.Tensor([length_lh]))
    else:
        length = torch.Tensor([feats.shape[1]])

    feats = feats.to(device)
    for model_idx in range(5):
        logits = models[model_idx](feats, length)
        ys = prediction2label(logits).cpu().tolist()
        y_pred.append(ys[0])
    return y_pred

def get_models(device, representation):
    models = []
    for model_idx in range(5):
        if representation == "p":
            model = PerformanceSummariser_ordinal(n_classes=9, input_size=88)
        elif representation == "virtuoso":
            model = PerformanceSummariser_ordinal(n_classes=9, input_size=64)
        else:
            model = PerformanceSummariser_argnn_ordinal(number_classes=9)
        model.load_state_dict(torch.load(f"eswa_difficulty/eswa_models/{representation}_ordinal-{model_idx}.pth"))
        model.to(device)
        model.eval()
        models.append(model)
    return models




def get_pitches(feats, compute_rh=True, compute_lh=True):
    rh = feats['rh']
    lh = feats['lh']
    rh_note_ids = rh["note_ids"].tolist()
    lh_note_ids = lh["note_ids"].tolist()
    rr, ll, embedding, onsets = 0, 0, [], []
    for ii in range(max(max(rh["note_ids"]), max(lh["note_ids"])) + 1):
        if compute_rh and ii in rh_note_ids:
            for _ in range(rh_note_ids.count(ii)):
                row_embedding = [0 for _ in range(88)]
                p = rh["pitches"][rr].tolist()
                if p*127 > 20 and p*127 <= 108:
                    row_embedding[int(p*127) - 21] = 1
                    o = rh["onsets"][rr].tolist()
                    onsets.append(o)
                rr += 1
                embedding.append(row_embedding)
        if compute_lh and ii in lh_note_ids:
            for _ in range(lh_note_ids.count(ii)):
                row_embedding = [0 for _ in range(88)]
                p = lh["pitches"][ll].tolist()
                if p*127 > 20 and p*127 <= 108:
                    row_embedding[int(p*127) - 21] = 1
                    o = lh["onsets"][ll].tolist()
                    onsets.append(o)
                ll += 1
                embedding.append(row_embedding)
    embedding = torch.Tensor(embedding)
    if compute_rh and compute_lh:
        assert len(rh_note_ids) + len(lh_note_ids) == int(embedding.shape[0])
    return embedding, onsets


def get_virtuoso():
    return torch.load(f'test.pt', map_location='cpu')['total_note_cat'].transpose(0, 1)


def get_argnn(embedding):
    new_embedding = torch_tuple(
        embedding['rh']['embedding']['lstm_ar_out'][0].squeeze(0),
        embedding['lh']['embedding']['lstm_ar_out'][0].squeeze(0)
    )
    return new_embedding


def average_difficulty(difficulty):
    # average difficulty if element is not -1
    difficulty = [x for x in difficulty if x != -1]
    return sum(difficulty) / len(difficulty) if len(difficulty) > 0 else -1


def compute_difficulty_pitch(path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feats_argnn = compute_embedding_score("ArGNNThumb-s", path)
    feats, _ = get_pitches(feats_argnn)
    models = get_models(device, 'p')
    difficulty = get_pred(feats, device=device, models=models, rep="p")
    return average_difficulty(difficulty)

def compute_difficulty_argnn(path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feats_argnn = compute_embedding_score("ArGNNThumb-s", path)
    feats = get_argnn(feats_argnn)
    models = get_models(device, 'argnn')
    difficulty = get_pred(feats, device=device, models=models, rep="argnn")
    return average_difficulty(difficulty)


def compute_difficulty_virtuoso(path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_path = path + ".pt"
    os.system(f"python3 eswa_difficulty/virtuosoNet/virtuoso --session_mode=inference "
              f"--checkpoint=eswa_difficulty/virtuosoNet/checkpoint_last.pt "
              f"--xml_path={path} --save_embedding={embedding_path}")
    emb = torch.load(embedding_path)['total_note_cat'].transpose(0, 1).squeeze()
    models = get_models(device, 'virtuoso')
    difficulty = get_pred(emb, device=device, models=models, rep="virtuoso")
    # remove embedding_path
    os.system(f"rm {embedding_path}")
    return average_difficulty(difficulty)


def compute_difficulty(path):
    diff_pitch = compute_difficulty_pitch(path)
    diff_argnn = compute_difficulty_argnn(path)
    diff_virtuoso = compute_difficulty_virtuoso(path)
    return average_difficulty([diff_pitch, diff_argnn, diff_virtuoso]), diff_pitch, diff_argnn, diff_virtuoso



if __name__ == '__main__':
    print(compute_difficulty(path="eswa_difficulty/test.xml"))