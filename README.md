# A simple Pytorch implementation of Differentiable Architecture Search (DARTS)

This repository is my pytorch implementation of [Differentiable Architecture Search (DARTS)](https://arxiv.org/abs/1806.09055).
Some of the code is taken from the [offical implementation](https://github.com/quark0/darts).    
 
## Requirements:
- python >= 3.5
- pytorch >= 1.0
- tensorboardX (optional)

## Search
``` 
python3 cifar_search.py --log_name darts_cifar_search --order 1st --gpus 0
```

## Evaluate
* ```python cifar_eval.py --log_name darts_cifar_search --gpus 0```
* ```python imgnet_eval.py --log_name darts_cifar_search --data_dir YOUR_IMGNET_DIR --gpus 0,1,2,3```

## CIFAR10 Results:
Method|Acc.|
:---:|:---:
DARTS 1st order|97.06%|
DARTS 2st order|97.36%|

