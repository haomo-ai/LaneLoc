
# Table of Contents

## 1. Introduction
We provide a training and test tutorials in this repository. 

We recommend you follow our code and data structures as follows.


## 2. Denpendecies

We use pytorch-gpu for neural networks.

An nvidia GPU is needed for faster retrival. OverlapTransformer is also fast enough when using the neural network on CPU.

To use a GPU, first you need to install the nvidia driver and CUDA.

## 3. Dataset introduction
Directory Structure:

```
TuSimple Ego-lane
  |
  |----train-valid/                   # video clips
         |----0313-1/                 # Sequential images for the clip, 20 frames
         |----0313-2
         |----0313-2
  |----test/                          # video clips
         |----0530/                   # Sequential images for the clip, 20 frames
         |----0601
```



```
CULane Ego-lane
  |
  |----train-valid/                   
         |----driver_23_30frames                
         |----driver_161_90frames 
         |----driver_182_30frame
  |----test/                          
         |----driver_37_30frames                   
         |----driver_100_30frames
         |----driver_193_90frames
```


## 4. How to use
### Step 1. Generate txt files  
根据前述步骤完成，原始数据下载、离线生成感知的车道线和道路边缘json文件  
运行 python3 data_util/data_gene_rgb.py，生成用于训练的txt文件
```
在19行设置数据root path  
如果想使用标注的车道线以及道路边缘结果，需要修改29行json目录  
```

### Step 2. Training  
Step.1 生成验证图片的path.txt

### Step 3. evaluation   
```
python evaluation.py
```
