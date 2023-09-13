
>Note: This code repository is in construction. Currently, only the dataset, the demo, and the inference code are available.


Dataset: https://zenodo.org/record/8037327

Demo: https://musiccritic.upf.edu/eswa_demo

# Combining piano performance dimensions for score difficulty classification

Pedro Ramoneda (a,∗), Dasaem Jeong (b), Vsevolod Eremenko (a), Nazif Can Tamer (a),
Marius Miron (a), Xavier Serra (a)
a: Universitat Pompeu Fabra, Barcelona, Spain
b: Sogang University, Seoul, Republic of Korea

Journal: Expert Systems with Applications (ESWA)

To cite this work, please use the following bibtex entry:

```
@article{ramoneda2023combining,
  title={Combining piano performance dimensions for score difficulty classification},
  author={Ramoneda, Pedro and Jeong, Dasaem and Eremenko, Vsevolod and Tamer, Nazif Can and Miron, Marius and Serra, Xavier},
  journal={Expert Systems with Applications (ESWA)},
  year={2023},
  address={a: Universitat Pompeu Fabra, Barcelona, Spain and b: Sogang University, Seoul, Republic of Korea}
}

```

## Abstract

Predicting the difficulty of playing a musical score is essential for structuring and exploring score collections. Despite its importance for music education, the automatic difficulty classification of piano scores is not yet solved, mainly due to the lack of annotated data and the subjectiveness of the annotations. This paper aims to advance the state-of-the-art in score difficulty classification with two major contributions. To address the lack of data, we present Can I Play It? (CIPI) dataset, a machine-readable piano score dataset with difficulty annotations obtained from the renowned classical music publisher Henle Verlag. 
The dataset is created by matching public domain scores with difficulty labels from Henle Verlag, then reviewed and corrected by an expert pianist.
As a second contribution, we explore various input representations from score information to pre-trained ML models for piano fingering and expressiveness inspired by the musicology definition of performance. We show that combining the outputs of multiple classifiers performs better than the classifiers on their own, pointing to the fact that the representations capture different aspects of difficulty. In addition, we conduct numerous experiments that lay a foundation for score difficulty classification and create a basis for future research. Our best-performing model reports a 39.5% balanced accuracy and 1.1 median square error across the nine difficulty levels proposed in this study. 
Code, dataset, and models are made available for reproducibility.

## Installation

```
pip install -r requirements.txt
```

## Inference

```python
from eswa_difficulty.compute_difficulty import compute_difficulty

path = 'path/to/musicxml'
diff_ensemble, diff_p, diff_argnn, diff_virtuoso = compute_difficulty(path)
```
