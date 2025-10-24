# Fine-tuning given Mole-BERT Pre-trained Model 

## Environment
The required packages are outputted into requirements.txt.

## Dataset
I include one example regression dataset ```esol``` and one example classification dataset ```bbbp``` for you to you test on. The rest of the molecule_net datasets can be downloaded from [chem data](http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip) (2.5GB), unzip it, and put it under `dataset/`. The two other regression ```malaria``` and ```cep``` datasets can be downloaded following the instruction [here](https://github.com/chao1224/GraphMVP/tree/main/datasets).


## Non Fewshot Fine-tuning (FT)
Some general arguments are ```dataset``` that can be chosen from ```esol, bbbp``` \
```split``` that can be chosen from ```random, scaffold, size```\
The range of model specific hyperparameters are specified in the appendix.

The baseline ```full FT``` can be run using:
```
python finetune_ood.py --dataset [dataset] --split [split] --tune_option all
```
The ```LP``` can be run using:
```
python finetune_ood.py --dataset [dataset] --split [split] --tune_option linear_layer
```
The ```surgical FT``` can be run using:
```
python finetune_ood.py --dataset [dataset] --split [split] --tune_option [tune_option]
```
The ```tune_option``` can be chosen from ```'first_linear', 'second_linear', 'third_linear', 'fourth_linear', 'fifth_linear'``` which indicating the GNN layer that will be updated during FT.

The ```LP-FT``` can be run using:
```
python lp_ft.py --dataset [dataset] --split [split]
```
The ```WiSE-FT``` can be run using:
```
python WISE.py --dataset [dataset] --split [split] --alpha [alpha]
```
where ```alpha``` is the mixing coefficient between [0,1]. You should first run the ```full FT``` to get the fully fine-tuned model for later weight interpolation. 

The ```L2-SP``` / ```BSS``` / ```Feature-map``` can be run using:
```
python reg_ft.py  --dataset [dataset] --split [split] --regularization_type [reg_type] --trade_off_backbone [delta] --trade_off_head [delta] --trade_off_bss [delta]
```
where ```reg_type``` can be chosen from ```'l2_sp, feature_map, bss'``` and ```delta``` is the regularization coefficient you can specify.

The ```DWiSE-FT``` can be run using:
```
python DWISE.py --dataset [dataset] --split [split] --epochs 200 --lr [lr] --alphas [alphas]
```
where ```lr``` is the learning rate we use; ```alphas``` is the initialization of alpha values to perform weight ensemble, where it should be in the form of ```"0.5_0.5_0.5_0.5_0.5_0.5"```, where you can specify different values to start with.

## Fewshot Fine-tuning (FT)
Fewshot FT uses the same commands as above, but with additional two arguments ```fewshot``` and ```fewshot_num```. For instance, you can run a full FT under fewshot setting with
```
python finetune_ood.py --dataset [dataset] --split [split] --tune_option all --fewshot True --fewshot_num [fewshot_num]
```
where ```fewshot_num``` can be chosen from ```50, 100, 500```

## Note
Note that before running ```WiSE-FT``` and ```DWiSE-FT```, you should first run the ```full FT``` to get the fully fine-tuned model for later weight interpolation. 


