# CycleGAN_Tensorlayer
Re-implement CycleGAN in Tensorlayer



#### Prerequisites:

* Tensorlayer
* TensorFlow
* Python

#### Run:
CUDA_VISIBLE_DEVICES=0 python main.py 

(if datasets are collected by yourself, you can use dataset_clean.py or dataset_crop.py to pre-process images)


#### Theory:

The generator process:

![Image text](https://github.com/luoxier/CycleGAN_Tensorlayer/blob/master/figures/generator.png "generator")  

The discriminator process:

![Image text](https://github.com/luoxier/CycleGAN_Tensorlayer/blob/master/figures/discriminator.png "discriminator")  

#### Results:

![Image text](https://github.com/luoxier/CycleGAN_Tensorlayer/blob/master/figures/result.png)  

![Image text](https://github.com/luoxier/CycleGAN_Tensorlayer/blob/master/figures/result2.png)  

#### Artifacts Remove

Using model_deconv:

![Image text](https://github.com/luoxier/CycleGAN_Tensorlayer/blob/master/figures/compare1.png) 

Using model_upsampling:

![Image text](https://github.com/luoxier/CycleGAN_Tensorlayer/blob/master/figures/compare2.png) 



#### Reference:

* Original Paper: https://arxiv.org/pdf/1703.10593.pdf
* Original implement in Torch: https://github.com/junyanz/CycleGAN/
* TensorLayer by HaoDong: https://github.com/zsdonghao/tensorlayer
* Resize Convolution: https://distill.pub/2016/deconv-checkerboard/

