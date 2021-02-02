# ANPyC
Source code for the paper:
[Overcoming Long-term Catastrophic Forgetting through
Adversarial Neural Pruning and Synaptic Consolidation]

## A tensorflow implementation of ANPyC on Permuted_MNIST.

## Requirement
python3.6
tensorflow 1.0 or later
linux

## Run
1. [Download MNIST](http://yann.lecun.com/exdb/mnist/)
2. run ANPyC without neural pruning, execute `python ANPyC_wo_NP.py`
3. run ANPyC without neural pruning, execute `python ANPyC_wi_NP.py`
4. run LwF, excute `python LwF.py`
5. run multi-task learning, excute `python joint.py`
6. run other baselines, i.e., ewc, mas, sgd with single head, sgd with freezing layers, fintuning, and SI, excute the simiar operation. 
7. cal the distribution of param-importance distribution, excute `python cal_importance_distribution.py`


###### This source code is released under a Attribution-NonCommercial 4.0 International license.

