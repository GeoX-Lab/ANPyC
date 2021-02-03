# ANPyC
This repo includes the source code for the paper:

Overcoming Long-term Catastrophic Forgetting through Adversarial Neural Pruning and Synaptic Consolidation


IEEE Transactions on Neural Networks and Learning Systems, 2021, 10.1109/TNNLS.2021.3056201

## Abstract:
Artificial neural networks face the well-known problem of catastrophic forgetting. What's worse, the degradation of previously learned skills becomes more severe as the task sequence increases, known as the long-term catastrophic forgetting. It is due to two facts: first, as the model learns more tasks, the intersection of the low-error parameter subspace satisfying for these tasks becomes smaller or even does not exist; second, when the model learns a new task, the cumulative error keeps increasing as the model tries to protect the parameter configuration of previous tasks from interference. Inspired by the memory consolidation mechanism in mammalian brains with synaptic plasticity, we propose a confrontation mechanism in which Adversarial Neural Pruning and synaptic Consolidation (ANPyC) is used to overcome the long-term catastrophic forgetting issue. The neural pruning acts as long-term depression to prune task-irrelevant parameters, while the novel synaptic consolidation acts as long-term potentiation to strengthen task-relevant parameters. During the training, this confrontation achieves a balance in that only crucial parameters remain, and non-significant parameters are freed to learn subsequent tasks. ANPyC avoids forgetting important information and makes the model efficient to learn a large number of tasks. Specifically, the neural pruning iteratively relaxes the current task's parameter conditions to expand the common parameter subspace of the task; the synaptic consolidation strategy, which consists of a structure-aware parameter-importance measurement and an element-wise parameter updating strategy, decreases the cumulative error when learning new tasks.


## A tensorflow implementation of ANPyC.

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

# The source code
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

Endnote
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
