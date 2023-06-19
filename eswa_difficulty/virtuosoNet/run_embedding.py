import json
import os
import subprocess

import torch


def load_json(name_file):
    data = None
    with open(name_file, 'r') as fp:
        data = json.load(fp)
    return data


def mikro():
    xml_path = 'Mikrokosmos-difficulty/musicxml'
    malas = 0
    malas_list = []
    for file_name in os.listdir(xml_path):
        try:
            print(file_name)
            os.system(f"python3 -m virtuoso --session_mode=inference --checkpoint=checkpoint_last.pt "
                      f"--xml_path={xml_path}/{file_name}")
            emb = torch.load(f'tmp.pt')
            print(emb["x"].shape)
            torch.save(emb, f"mikrokosmos_embedding/{file_name[:-4]}.pt")
        except Exception as e:
            malas += 1
            malas_list.append(file_name)
            print(e)
    print(malas, malas_list)


def henle():
    base_path = f'henle_embedding'
    malas = 0
    block = []
    malas_list = []
    the_data = load_json("CIPI_symbolic/index.json")
    for key in reversed(list(the_data.keys())):
        if os.path.exists(f'CIPI_symbolic/virtuoso/{key}.pt'):

            print(f"computado {key}.pt")
            continue
        try:
            print(key)
            note_f = []
            beat_f = []
            measure_f = []
            beat_spanned_f = []
            measure_spanned_f = []
            total_note_cat_f = []
            x_f = []
            for path in the_data[key]['path'].values():
                my_timeout = 240
                p = subprocess.Popen([
                    'python3', '-m', 'virtuoso', '--session_mode=inference', '--checkpoint=checkpoint_last.pt',
                    '--device=cpu', f"--xml_path={'CIPI_symbolic/' + path}"
                ])
                p.wait(my_timeout)
                if p.returncode != 0:
                    malas += 1
                    print("bad read")
                    raise Exception
                else:
                    emb = torch.load(f'tmp.pt')
                    note_f.append(emb['note'])
                    beat_f.append(emb['beat'])
                    measure_f.append(emb['measure'])
                    beat_spanned_f.append(emb['beat_spanned'])
                    measure_spanned_f.append(emb['measure_spanned'])
                    total_note_cat_f.append(emb['total_note_cat'])
                    x_f.append(emb['x'])

            final_embedding = {}
            final_embedding["note"] = torch.concat(note_f, dim=1)
            final_embedding["beat"] = torch.concat(beat_f, dim=1)
            final_embedding["measure"] = torch.concat(measure_f, dim=1)
            final_embedding["beat_spanned"] = torch.concat(beat_spanned_f, dim=1)
            final_embedding["measure_spanned"] = torch.concat(measure_spanned_f, dim=1)
            final_embedding["total_note_cat"] = torch.concat(total_note_cat_f, dim=1)
            final_embedding["x"] = torch.concat(x_f, dim=1)
            torch.save(final_embedding, f"CIPI_symbolic/virtuoso/{key}.pt")
        except Exception as e:
            malas_list.append(key)
            print("ha saltado la excepcion")
            print(e)
            #raise e


    print(malas, malas_list, block, sep="\n")


if __name__ == '__main__':
    os.system(f"python3 -m virtuoso --session_mode=inference --checkpoint=checkpoint_last.pt "
              f"--xml_path=../test.xml --save_embedding=tmp.pt")
    emb = torch.load(f'tmp.pt')