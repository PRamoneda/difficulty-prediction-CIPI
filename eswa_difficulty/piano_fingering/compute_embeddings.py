import traceback

import numpy as np
import torch

from eswa_difficulty.piano_fingering import GGCN
from eswa_difficulty.piano_fingering import seq2seq_model, common
from eswa_difficulty.virtuosoNet.virtuoso.pyScoreParser.data_class import ScoreData


def choice_model(hand, architecture):
    # load model torch implementation

    model = None
    if architecture == 'ArGNNThumb-s':
        model = seq2seq_model.seq2seq(
            embedding=common.emb_pitch(),
            encoder=seq2seq_model.gnn_encoder(input_size=64),
            decoder=seq2seq_model.AR_decoder(64)
        )
    elif architecture == 'ArLSTMThumb-f':
        model = seq2seq_model.seq2seq(
            embedding=common.only_pitch(),
            encoder=seq2seq_model.lstm_encoder(input=1, dropout=0),
            decoder=seq2seq_model.AR_decoder(64)
        )
    if model is not None:
        assert model is not None, "bad model chosen"
    # load model saved from checkpoint
    model_path = f"{hand}_{architecture}.pth"
    checkpoint = torch.load(f'eswa_difficulty/piano_fingering/models/{model_path}', map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def next_onset(onset, onsets):
    # -1 is a impossible value then there is no next
    ans = '-1'
    hand_onsets = list(set(onsets))
    hand_onsets.sort(key=lambda a: float(a))
    for idx in range(len(hand_onsets)):
        if float(hand_onsets[idx]) > float(onset):
            ans = hand_onsets[idx]
            break
    return ans


def compute_edge_list(onsets, pitchs):
    assert len(onsets) == len(pitchs), "check lenghts"
    edges = []
    for idx, (current_onset, current_pitch) in enumerate(zip(onsets, pitchs)):
        # pdb.set_trace()
        if current_pitch != 0:
            # next labels of right hand
            next_right_hand = next_onset(current_onset, onsets)
            next_labels = [(idx, jdx, "next") for jdx, onset in enumerate(onsets) if
                           onset == next_right_hand and idx != jdx]
            edges.extend(next_labels)
            # onset labels
            onset_edges = [(idx, jdx, "onset") for jdx, onset in enumerate(onsets) if
                           current_onset == onset and idx != jdx]
            edges.extend(onset_edges)
    return edges


def first_note_symmetric(note, from_hand='lh'):
    right2left_pitch_class_symmetric = {
        0: 4,
        1: 2,
        2: 0,
        3: -2,
        4: -4,
        5: -6,
        6: -8,
        7: -10,
        8: -12,
        9: -14,
        10: -16,
        11: -18
    }
    left2right_pitch_class_symmetric = {
        0: 16,
        1: 14,
        2: 12,
        3: 10,
        4: 8,
        5: 6,
        6: 4,
        7: 2,
        8: 0,
        9: -2,
        10: -4,
        11: -6
    }
    pitch_class = note % 12  # 4
    d_oct = (note - 60) // 12  # -1

    if from_hand == 'lh':
        ans = note + left2right_pitch_class_symmetric[pitch_class] - (2 * d_oct * 12) - 24
    else:
        ans = note + right2left_pitch_class_symmetric[pitch_class] - (2 * d_oct * 12)
    return ans

def _surpass_bounds(notes):
    surpass = False
    for n in notes:
        if not (n == 0 or (21 <= n < 108)):
            surpass = True
    return surpass


def reverse_hand(data, bounds=False):
    list_notes, list_onsets, list_durations, list_fingers, list_ids, list_lengths, list_edges = [], [], [], [], [], [], []
    for notes, onsets, durations, fingers, ids, lengths, edges in zip(*data):
        new_notes = []
        notes = notes * 127
        jdx = 0
        for idx, n in enumerate(notes):
            if n == 0:
                jdx += 1
                new_notes.append(0)
            elif idx == jdx:
                new_notes.append(first_note_symmetric(notes[idx]))
            else:
                is_black_current = (n % 12) in [1, 3, 6, 8, 10]
                distance = n - notes[idx - 1]
                new_n = new_notes[-1] - distance
                is_black_new = (new_n % 12) in [1, 3, 6, 8, 10]
                new_notes.append(new_n)
                assert is_black_current == is_black_new, " is not working symmetric hand data augmentation " \
                                                         f"original seq = {np.array(notes)} " \
                                                         f"new seq = {np.array(new_notes)}"

        new_notes = np.array(new_notes)
        if bounds:
            if _surpass_bounds(new_notes):
                print(f"surpass piano keyboard bounds "
                      f"original seq = {np.array(notes)} "
                      f"new seq = {np.array(new_notes)}")
                continue
        list_notes.append(new_notes / 127)
        list_onsets.append(onsets)
        list_durations.append(durations)
        list_fingers.append(fingers)
        list_ids.append(ids)
        list_lengths.append(lengths)
        list_edges.append(edges)
    return list_notes, list_onsets, list_durations, list_fingers, list_ids, list_lengths, list_edges

def predict_score(model, pitches, onsets, durations, hand):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if hand == 'lh':
        data = [pitches], [onsets], [durations], [0], [0], [0], [0]
        data = reverse_hand(data)
        pitches = data[0][0]
    edges = compute_edge_list(onsets, pitches)
    edges = edges_to_matrix(edges, len(pitches))
    pitches = torch.Tensor(pitches).view(1, -1, 1)
    onsets = torch.Tensor(onsets).view(1, -1, 1)
    durations = torch.Tensor(durations).view(1, -1, 1)

    model.to(device)
    # print(len(pitches.shape))
    model.eval()
    with torch.no_grad():
        out, embedding = model.get_embedding(pitches.to(device),
                                             onsets.to(device),
                                             durations.to(device),
                                             torch.Tensor([pitches.shape[1]]).to(device),
                                             edges.to(device),
                                             None, beam_k=6)
    fingers_piece = [x + 1 for x in out[0]]
    if hand == 'lh':
        fingers_piece = [-1 * ff for ff in fingers_piece]
    return fingers_piece, embedding

def edges_to_matrix(edges, num_notes, graph_keys=GGCN.GRAPH_KEYS):
    if len(graph_keys) == 0:
        return None
    num_keywords = len(graph_keys)
    graph_dict = {key: i for i, key in enumerate(graph_keys)}
    if 'rest_as_forward' in graph_dict:
        graph_dict['rest_as_forward'] = graph_dict['forward']
        num_keywords -= 1
    matrix = np.zeros((num_keywords * 2, num_notes, num_notes))
    edg_indices = [(graph_dict[edg[2]], edg[0], edg[1])
                   for edg in edges
                   if edg[2] in graph_dict]
    reverse_indices = [(edg[0] + num_keywords, edg[2], edg[1]) if edg[0] != 0 else
                       (edg[0], edg[2], edg[1]) for edg in edg_indices]
    edg_indices = np.asarray(edg_indices + reverse_indices)

    matrix[edg_indices[:, 0], edg_indices[:, 1], edg_indices[:, 2]] = 1

    matrix[num_keywords, :, :] = np.identity(num_notes)
    matrix = torch.Tensor(matrix)
    return matrix


def isAscending(list):
    previous = list[0]
    for number in list:
        if number < previous:
            return False
        previous = number
    return True


def compute_embedding_score(architecture, path):
    model = {
        'lh': choice_model('lh', architecture),
        'rh': choice_model('rh', architecture)
    }
    result = {}
    try:
        sc = ScoreData(f"{path}", None, 'Mozart', read_xml_only=True)
        om = [{
            "note_number": number_note,
            "midi": note.pitch[1],
            "offsetSeconds": note.note_duration.xml_position,
            "hand": 'rh' if note.voice < 5 else 'lh',
            "measure_number": note.measure_number}
            for number_note, note in enumerate(sc.xml_notes)
        ]
        for hand in ['rh', 'lh']:
            onsets = np.array([o['offsetSeconds'] for o in om if o['hand'] == hand])
            pitches = np.array([((int(o['midi'])) / 127.0) for o in om if o['hand'] == hand])
            note_numbers = np.array([o['note_number'] for o in om if o['hand'] == hand])

            fingers, embedding = predict_score(
                model[hand],
                pitches,
                onsets,
                [],
                hand,
            )
            result[hand] = {"note_ids": note_numbers, "embedding": embedding, "pitches": pitches, "onsets": onsets}

    except Exception as e:
        print("Meehhh error")
        print(e)
        traceback.print_exc()
    return result


if __name__ == '__main__':
    result = compute_embedding_score("ArGNNThumb-s", "eswa_difficulty/test.xml")
    print(result)

