# pytorch_ano_pre
Pytorch Re-implemention of ano_pre_cvpr2018, replace flownet2 with lite-flownet

![img](https://github.com/StevenLiuWen/ano_pred_cvpr2018/blob/master/assets/architecture.JPG)

[Future Frame Prediction for Anomaly Detection -- A New Baseline, CVPR 2018](https://arxiv.org/pdf/1712.09867.pdf)

[tensorflow_offical_implement](https://github.com/StevenLiuWen/ano_pred_cvpr2018)

This Implementation is much easier to converge than the original repo, which we take 20000 steps while the tf one take 80000 steps with AUC 84.9%, and get an outperform result with AUC of __85.5%+-0.75%__, which use no flow_loss.

__It's interesting that the result of this repo works better without flow loss than the paper said, which said that without flow_loss the AUC is *83.7%* __

![img](https://github.com/fjchange/pytorch_ano_pre/blob/master/Assests/Screenshot%20from%202019-04-06%2023-06-49.png)
![img](https://github.com/fjchange/pytorch_ano_pre/blob/master/Assests/Screenshot%20from%202019-04-06%2023-02-31.png)
![img](https://github.com/fjchange/pytorch_ano_pre/blob/master/Assests/Screenshot%20from%202019-04-06%2023-02-54.png)

## 1. requirement
- pytorch >=0.4.1
- tensorboardX (if you want)

## 2. preparation
1. Download Dataset CUHK Avenue [download_link](https://onedrive.live.com/?authkey=%21AMqh2fTSemfrokE&id=3705E349C336415F%215109&cid=3705E349C336415F), unzip in the path you want, and replace the path in **train.py**

2. Download Lite-Flownet model, and replace the path in **train.py**
> wget --timestamping http://content.sniklaus.com/github/pytorch-liteflownet/network-default.pytorch

3. replace all the modle_output_path and log_output_path to where you want in **train.py**

## 3. training

> cd ano_pre

> python train.py

## 4. evalute
replace the model_path and evaluate_name as you want

> cd ano_pre

> python evaluate.py

![img](https://github.com/fjchange/pytorch_ano_pre/blob/master/Assests/image.png)

## 5. reference

If you find this useful, please cite the work as follows:

```code
[1]  @INPROCEEDINGS{liu2018ano_pred, 
        author={W. Liu and W. Luo, D. Lian and S. Gao}, 
        booktitle={2018 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
        title={Future Frame Prediction for Anomaly Detection -- A New Baseline}, 
        year={2018}   
     }   
[2]  misc{pytorch_ano_pred,
          author = {Jiachang Feng},
          title = { A Reimplementation of {Ano_pred} Using {Pytorch}},
          year = {2019},
          howpublished = {\url{https://github.com/fjchange/pytorch_ano_pre}}    
    }
[3]  @inproceedings{Hui_CVPR_2018,
         author = {Tak-Wai Hui and Xiaoou Tang and Chen Change Loy},
         title = {{LiteFlowNet}: A Lightweight Convolutional Neural Network for Optical Flow Estimation},
         booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
         year = {2018}  
     }
[4]  @misc{pytorch-liteflownet,
         author = {Simon Niklaus},
         title = {A Reimplementation of {LiteFlowNet} Using {PyTorch}},
         year = {2019},
         howpublished = {\url{https://github.com/sniklaus/pytorch-liteflownet}}      
    }
```
