<h1 align="center">RoFt-Mol: Benchmarking Robust Fine-Tuning with Molecular Graph Foundation Models (RoFT-MoL)</h1>

This repository is the implementation for the paper [RoFt-Mol: Benchmarking Robust Fine-Tuning with Molecular Graph Foundation Models](https://arxiv.org/abs/2509.00614) by Shikun Liu, Deyu Zou, Nima Shoghi, Victor Fung, Kai Liu, Pan Li.

## Overview

We introduce **RoFt-Mol**, which benchmarks robust fine-tuning methods on molecular property prediction tasks given pre-trained molecular graph foundation models to better understand how to fine-tune the pre-trained models to the downstream tasks robustly under limited samples and potential distribution shifts. We classify eight fine-tuning methods into three mechanisms: weight-based, representation-based, and partial fine-tuning and benchmark these methods on downstream regression and classification tasks across supervised and self-supervised pre-trained models in diverse labeling settings. This extensive evaluation provides valuable insights and informs the design of a refined robust fine-tuning method, ROFT-MOL. This approach combines the strengths of
simple post-hoc weight interpolation with more complex weight ensemble finetuning methods, delivering improved performance across both task types while maintaining the ease of use inherent in post-hoc weight interpolation. 

## Pretrained Models

We include total 6 pretrained models and benchmark the fine-tuning methods on top of them. You can refer to the README file inside each pretrained model to see how to run the fine-tuning method in detail for that specific model. Here, the graphium folder refers to both the graphium-toy and graphium-large in our paper. 

## Downstream Datasets

We utilize the standard MoleculeNet datasets as our downstream task evaluations, we include 8 classification tasks and 4 regression tasks. 

The molecule_net datasets can be downloaded from [chem data](http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip) (2.5GB), unzip it, and put it under `dataset/` for each pre-trained model. The two other regression ```malaria``` and ```cep``` datasets can be downloaded following the instruction [here](https://github.com/chao1224/GraphMVP/tree/main/datasets).

## Acknowledgements
This repository inherits the code from [Mole-BERT](https://github.com/junxia97/Mole-BERT), [Graphium](https://github.com/datamol-io/graphium), [GraphGPS](https://github.com/rampasek/GraphGPS), [GraphMAE](https://github.com/THUDM/GraphMAE) and [MoleculeSTM](https://github.com/chao1224/MoleculeSTM). 

## Reference
```
@article{liu2025roft,
  title={RoFt-Mol: Benchmarking Robust Fine-Tuning with Molecular Graph Foundation Models},
  author={Liu, Shikun and Zou, Deyu and Shoghi, Nima and Fung, Victor and Liu, Kai and Li, Pan},
  journal={arXiv preprint arXiv:2509.00614},
  year={2025}
}
```

