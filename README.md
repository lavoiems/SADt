# SPUDT

Official code repository for the paper Semantic-Preserving Unsupervised Domain Translation.

Dependencies are provided in `requirement.txt`. A Dockerfile is provided for reproducing the same environment used to run the experiments.

## Organization

The models are in `src/models`. Every model has 3 files:

`__init__.py`: Defines the specific parameters of the models

`model.py` Defines the architecture of the model

`train.py` Defines the training algorithm of the model

## Run model

A model can be run by invoking `main.py`. The syntax is as follows
```
python src/main.py [GENERAL PARAMETERS] [MODEL] [SPECIFIC MODEL PARAMETERS]
```

As example, we provide the command for reproducing our results on MNIST to SVHN

### Clustering
**MNIST**
```
python src/main.py --exp-name cluster --cuda --run-id mnist vrinv --dataset1 imnist --lr 1e-3 --h-dim 256 --ld 1
```

### Domain adaptation with clustering

**MNIST-to-SVHN**
```
python src/main.py --exp-name vmt-cluster --cuda --run-id mnist-svhn vmt_cluster --dataset1 mnist --dataset2 svhn --cluster-model-path ./experiments/vrinv/cluster_mnist-None --cluster-model vrinv --dw 0.01 --svw 1 --tvw 0.06 --tcw 0.06 --smw 1 --tmw 0.06
```


### Classifier for UDT evaluation

**SVHN** 
```
python src/main.py --exp-name classifier --cuda --run-id svhn --train-batch-size 128 --valid-split 0.2 classifier --dataset svhn_extra
```

### Conditional generation

**MNIST-to-SVHN**
```
python src/main.py --run-id mnist-svhn --exp-name UDT --train-batch-size 64 --test-batch-size 50 udt --eval-model-path ./experiments/classifier/classifier-wide_mnist-None/ --dataset1 mnist --dataset2 svhn --semantic-model-path ./experiments/vmt_cluster/vmt-cluster_mnist-svhn-None --gsxy 0.5
```

## Visualizing the results
We use tensorboard for saving the artefacts. It is possible to view the results by simply invoking tensorboard in the folder where the results were saved
```
tensorboard --logdir .
```
