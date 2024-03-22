# SFedFR_Example
The code of SFedFR method


## 结构
Tutorials：运行代码

datasets：数据集

fedlab：框架代码和算法代码

partiton-reports：中间结果保存

## 运行
```shell
#! /bin/bash

for seed in 0 10 100; do
  echo "Running code with random seed: $seed"
  python mypoc_mnist_SelectUpdate.py $seed
done
```