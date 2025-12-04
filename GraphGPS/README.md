# Fine-tuning given GraphGPS Pre-trained Model 

## Environment
The required packages are outputted into requirements.txt.

## Pretrained Model
Please download the pretrained model following the instruction in the original GraphGPS repo
```
wget https://www.dropbox.com/s/aomimvak4gb6et3/pcqm4m-GPS%2BRWSE.deep.zip
unzip pcqm4m-GPS+RWSE.deep.zip -d pretrained/
```

## Some general parameters to run the fine-tuning
Some general arguments are ```dataset``` that can be chosen from the list of 12 datasets \
```split``` that can be chosen from ```random, scaffold, size```\
```fewshot``` is a boolean indicating whether we are using fewshot samples for fine-tuning
```fewshot_num``` will be the number of samples we use if we do fewshot samples, it can be chosen from ```50, 100, 500```
The range of model specific hyperparameters are specified in the appendix.


## Specific configurations to run each method
The baseline ```full FT``` can be run using:
```
python main_ft.py --cfg configs/GPS/[dataset]_ft_full.yaml --repeat 5 [split] [fewshot] [fewshot_num] 0.001(dummy coefficient that not being used)
```
The ```LP``` can be run using:
```
python main_ft.py --cfg configs/GPS/[dataset]_ft_lp.yaml --repeat 5 [split] [fewshot] [fewshot_num] 0.001(dummy coefficient that not being used)
```
The ```surgical FT``` can be run using:
```
python main_ft.py --cfg configs/GPS/[dataset]_ft_surg.yaml --repeat 5 [split] [fewshot] [fewshot_num] [tune_layer]
```
where ```tune_layer``` can be chosen from ```'front', 'mid', 'back', which indicating the GNN layers that will be updated during FT.

The ```LP-FT``` can be run using:
```
python main_ft.py --cfg configs/GPS/[dataset]_ft_lpft.yaml --repeat 5 [split] [fewshot] [fewshot_num] 0.001(dummy coefficient that not being used)
```
The ```WiSE-FT``` can be run using:
```
python wise_ft.py --cfg configs/GPS/[dataset]_ft_full.yaml --repeat 1 [split] [fewshot] [fewshot_num] [alpha]
```
where ```alpha``` is the mixing coefficient between [0,1]. You should first run the ```full FT``` to get the fully fine-tuned model for later weight interpolation. 

The ```L2-SP``` can be run using:
```
python main_ft.py --cfg configs/GPS/[dataset]_ft.yaml --repeat 5 [split] [fewshot] [fewshot_num] [delta]
```
where ```delta``` is the regularization coefficient you can specify.

The ```BSS``` can be run using:
```
python main_ft.py --cfg configs/GPS/[dataset]_ft_bss.yaml --repeat 5 [split] [fewshot] [fewshot_num] [delta]
```
where ```delta``` is the regularization coefficient you can specify.

The ```Feature_map``` can be run using:
```
python main_ft.py --cfg configs/GPS/[dataset]_ft_fm.yaml --repeat 5 [split] [fewshot] [fewshot_num] [delta]
```
where ```delta``` is the regularization coefficient you can specify.

## Note
Note that before running ```WiSE-FT```, you should first run the ```full FT``` to get the fully fine-tuned model for later weight interpolation. 


