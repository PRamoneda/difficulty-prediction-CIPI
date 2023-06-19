import os

from eswa_difficulty.compute_difficulty import compute_difficulty

for path in os.listdir('examples'):
    path = f'examples/{path}'
    print(path)
    diff_ensemble, diff_p, diff_argnn, diff_virtuoso = compute_difficulty(path)
    print(f"{path} & {diff_ensemble:.2f} & {diff_p:.2f} & {diff_argnn:.2f} & {diff_virtuoso:.2f} \\\\")
