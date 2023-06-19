import json
import pickle

import music21
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, mean_squared_error
import json
import numpy as np

KEY_TO_SEMITONE = {'c': 0, 'c#': 1, 'db': 1, 'd-': 1, 'c##': 2, 'd': 2, 'e--': 2, 'd#': 3, 'eb': 3, 'e-': 3, 'd##': 4,
                   'e': 4, 'f-': 4, 'e#': 5, 'f': 5, 'g--': 5, 'e##': 6, 'f#': 6, 'gb': 6, 'g-': 6, 'f##': 7, 'g': 7,
                   'a--': 7, 'g#': 8, 'ab': 8, 'a-': 8, 'g##': 9, 'a': 9, 'b--': 9, 'a#': 10, 'bb': 10, 'b-': 10,
                   'a##': 11, 'b': 11, 'b#': 12, 'c-': -1, 'x': None}

def prediction2label(pred):
    """Convert ordinal predictions to class labels, e.g.

    [0.9, 0.1, 0.1, 0.1] -> 0
    [0.9, 0.9, 0.1, 0.1] -> 1
    [0.9, 0.9, 0.9, 0.1] -> 2
    etc.
    """
    return (pred > 0.5).cumprod(axis=1).sum(axis=1) - 1


def compute_metrics(model, name_subset, X_subset, y_subset):
    pred = model.predict(X_subset)

    bacc = balanced_accuracy_score(y_pred=pred, y_true=y_subset)

    mse = mean_squared_error(y_pred=pred, y_true=y_subset)

    mask = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    bacc3 = balanced_accuracy_score(y_pred=[mask[yy] for yy in pred], y_true=[mask[yy] for yy in y_subset])

    matches = [1 if pp in [tt - 1, tt, tt + 1] else 0 for tt, pp in zip(y_subset, pred)]
    acc_plusless_1 = sum(matches) / len(matches)

    print(json.dumps(
        {
            f'bacc-{name_subset}': bacc,
            f'3bacc-{name_subset}': bacc3,
            f'acc_plusless_1-{name_subset}': acc_plusless_1,
            f'mse-{name_subset}': mse
        }, indent=4)
    )


def save_json(dictionary, name_file):
    with open(name_file, 'w') as fp:
        json.dump(dictionary, fp, sort_keys=True, indent=4)


def load_json(name_file):
    data = None
    with open(name_file, 'r') as fp:
        data = json.load(fp)
    return data


def save_binary(dictionary, name_file):
    with open(name_file, 'wb') as fp:
        pickle.dump(dictionary, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load_binary(name_file):
    data = None
    with open(name_file, 'rb') as fp:
        data = pickle.load(fp)
    return data


def strm2map(strm, hand):
    strm = strm.parts[1] if hand == 'lh' else strm.parts[0]
    converted = []
    om = []
    for o in strm.flat.secondsMap:
        if o['element'].isClassOrSubclass(('Note',)):
            finger = [art.fingerNumber for art in o["element"].articulations if
                      type(art) == music21.articulations.Fingering]
            if len(finger) == 1 and finger[0] in [1, 2, 3, 4, 5]:
                o['finger'] = finger[0]
            else:
                o['finger'] = 0
            om.append(o)
        elif o['element'].isClassOrSubclass(('Chord',)):
            articulations = [
                art.fingerNumber for art in o["element"].articulations
                if type(art) == music21.articulations.Fingering and art.fingerNumber in [0, 1, 2, 3, 4, 5]
            ]
            if len(articulations) == len(o['element']):
                if hand == 'lh':
                    fingers = list(sorted(articulations, reverse=True))
                else:
                    fingers = list(sorted(articulations))
            else:
                fingers = [0] * len(o['element'])

            om_chord = [
                {
                    'element': oc,
                    'offsetSeconds': o['offsetSeconds'],
                    'endTimeSeconds': o['endTimeSeconds'],
                    'chord': o['element'],
                    'finger': finger
                }
                for oc, finger in zip(sorted(o['element'].notes, key=lambda a: a.pitch), fingers)
            ]
            om.extend(om_chord)
    om_filtered = []
    for o in om:
        offset = o['offsetSeconds']
        duration = o['endTimeSeconds']
        pitch = o['element'].pitch
        simultaneous_notes = [o2 for o2 in om if
                              o2['offsetSeconds'] == offset and o2['element'].pitch.midi == pitch.midi]
        max_duration = max([float(x['endTimeSeconds']) for x in simultaneous_notes])
        if len(simultaneous_notes) > 1 and duration < max_duration and str(offset) + ':' + str(pitch) not in converted:
            continue
        else:
            converted.append(str(offset) + ':' + str(pitch))

        if not (o['element'].tie and (o['element'].tie.type == 'continue' or o['element'].tie.type == 'stop')) and \
                not ((hasattr(o['element'], 'tie') and o['element'].tie
                      and (o['element'].tie.type == 'continue' or o['element'].tie.type == 'stop'))) and \
                not (o['element'].duration.quarterLength == 0):
            om_filtered.append(o)

    return sorted(om_filtered, key=lambda a: (a['offsetSeconds'], a['element'].pitch))