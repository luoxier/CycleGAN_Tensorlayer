# CycleGAN_Tensorlayer
Re-implement CycleGAN in TensorLayer

- Original CycleGAN
- Improved CycleGAN with resize-convolution



### Prerequisites:

* TensorLayer
* TensorFlow
* Python

### Run:
```Bash
CUDA_VISIBLE_DEVICES=0 python main.py 
```

(if datasets are collected by yourself, you can use dataset_clean.py or dataset_crop.py to pre-process images)


### Theory:

The generator process:

![Image text](https://github.com/luoxier/CycleGAN_Tensorlayer/blob/master/figures/generator.png "generator")  

The discriminator process:

![Image text](https://github.com/luoxier/CycleGAN_Tensorlayer/blob/master/figures/discriminator.png "discriminator")  

 

### Result Improvement
* Data augmentation
* Resize convolution[4]
* Instance normalization[5]



#### data augmentation:
![Image text](https://github.com/luoxier/CycleGAN_Tensorlayer/blob/master/figures/data_augmentation.png) 


#### Instance normalization（comparision by original paper https://arxiv.org/abs/1607.08022）:
![Image text](https://github.com/luoxier/CycleGAN_Tensorlayer/blob/master/figures/instance_norm.png) 


#### Resize convolution (Remove Checkerboard Artifacts):

![Image text](https://github.com/luoxier/CycleGAN_Tensorlayer/blob/master/figures/compare1.png) 

![Image text](https://github.com/luoxier/CycleGAN_Tensorlayer/blob/master/figures/compare2.png) 

### Final Results:

![Image text](https://github.com/luoxier/CycleGAN_Tensorlayer/blob/master/figures/result.png)  

![Image text](https://github.com/luoxier/CycleGAN_Tensorlayer/blob/master/figures/result2.png) 



### Reference:

* [1] Original Paper: https://arxiv.org/pdf/1703.10593.pdf
* [2] Original implement in Torch: https://github.com/junyanz/CycleGAN/
* [3] TensorLayer by HaoDong: https://github.com/zsdonghao/tensorlayer
* [4] Resize Convolution: https://distill.pub/2016/deconv-checkerboard/
* [5] Instance Normalization: https://arxiv.org/abs/1607.08022

