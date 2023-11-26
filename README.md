
# Table of Contents

## 1. Introduction
We provide a training and test tutorials in this repository. 

We recommend you follow our code and data structures as follows.


## 2. Denpendecies

We use pytorch-gpu for neural networks.

An nvidia GPU is needed for faster retrival. LaneLoc is also fast enough when using the neural network on CPU.

To use a GPU, first you need to install the nvidia driver and CUDA.

## 3. Dataset introduction

The ego-lane index annotation results can be downloaded from: 

https://drive.google.com/file/d/1HwxNsma9yj4ZNvZ2vIXjMAS4w50LVCJJ/view?usp=sharing
https://drive.google.com/file/d/1CTZCoQWQ_zKXqk0DYjT6aGSVnyGo5oCY/view?usp=sharing

the dataset should be organized by this Structure:
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
generate the txt files for training
```
python txt_tusimple.py
python txt_culane.py
```

### Step 2. run demo  
```
python demo_culane.py
python demo_tusimple.py
```
