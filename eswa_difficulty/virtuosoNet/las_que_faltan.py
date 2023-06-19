


import os


for p in ['2364046', '5026044', '5026820', '5028687', '5047635', '5074210', '5112058', '5187218', '5452009']:
    os.system(f"python3 -m virtuoso --session_mode=inference --checkpoint=checkpoint_last.pt "
              f"--device=cpu --xml_path=henleXmus/score_files/{p}.musicxml")