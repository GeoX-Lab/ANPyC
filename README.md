# ANPyC
Source code for the paper:
[Overcoming Long-term Catastrophic Forgetting through
Adversarial Neural Pruning and Synaptic Consolidation]()

## A tensorflow implementation of ANPyC on Permuted_MNIST.

### Requirement

python3.6

tensorflow 1.0 or later

linux

### Run

1. [Download MNIST](http://yann.lecun.com/exdb/mnist/)

2. run ANPyC without neural pruning by `python ANPyC_wo_NP.py`

3. run ANPyC with neural pruning by `python ANPyC_wi_NP.py`

4. run LwF by `python LwF.py`

5. run multi-task learning by `python joint.py`

6. run other baselines, i.e., ewc, mas, sgd with single head, sgd with freezing layers, fintuning, and SI by the similar operation. 

7. cal the distribution of param-importance distribution by `python cal_importance_distribution.py`



This source code is released under a Attribution-NonCommercial 4.0 International license, find more about it [here](https://github.com/GeoX-Lab/ANPyC/blob/main/LICENSE)

# ANPyC
If our repo is useful to you, please cite our published paper as follow:

```Bibtex
@article{peng2021ANPyc,
    title={Overcoming Long-term Catastrophic Forgetting through Adversarial Neural Pruning and Synaptic Consolidation},
    author={Peng, Jian and Tang, Bo and Jiang, Hao and Li, Zhuo and Lei, Yinjie and Lin, Tao and Li, Haifeng},
    journal={IEEE Transactions on Neural Networks and Learning Systems},
    DOI = {10.1109/TNNLS.2021.3056201},
    year={2021},
    type = {Journal Article}
}

```Endnote
%0 Journal Article
%A Peng, Jian
%A Tang, Bo
%A Jiang, Hao
%A Li, Zhuo
%A Lei, Yinjie
%A Lin, Tao
%A Li, Haifeng
%D 2021
%T Overcoming Long-term Catastrophic Forgetting through Adversarial Neural Pruning and Synaptic Consolidation
%B IEEE Transactions on Neural Networks and Learning Systems
%R 10.1109/TNNLS.2021.3056201
%! Overcoming Long-term Catastrophic Forgetting through Adversarial Neural Pruning and Synaptic Consolidation
