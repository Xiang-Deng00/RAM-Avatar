<div align="center">

# <b>RAM-Avatar: Real-time Photo-Realistic Avatar from Monocular Videos </b>with Full-body Control
Xiang Deng<sup>1</sup>, [Zerong Zheng](https://zhengzerong.github.io/)<sup>2</sup>, [Yuxiang Zhang](https://zhangyux15.github.io/)<sup>1</sup>, [Jingxiang Sun](https://mrtornado24.github.io/)<sup>1</sup>, Chao Xu<sup>2</sup>, XiaoDong Yang<sup>3</sup>, [Lizhen Wang](https://lizhenwangt.github.io/)<sup>1</sup>, [Yebin Liu](https://www.liuyebin.com)<sup>1</sup>


<sup>1</sup>Tsinghua Univserity  <sup>2</sup>NNKosmos Technology  <sup>3</sup>Li Auto

###  [Paper (Early access)](https://cloud.tsinghua.edu.cn/f/6b7a88c3b4ac43b0b506/?dl=1) · [Video]

</div>
<img src="https://github.com/Xiang-Deng00/RAM-Avatar/blob/main/sample_results.png">

***Abstract**: This paper focuses on advancing the applicability of human avatar learning methods by proposing RAM-Avatar, which learns Real-time, photo-realistic Avatar supports full-body control from Monocular videos. To achieve this goal, RAM-Avatar leverages two statistical templates responsible for modeling the facial expression and hand gesture variations, while a sparsely computed dual attention module is introduced upon another body template to facilitate high-fidelity texture rendering for the torsos and limbs. Building on this foundation, we deploy a lightweight yet powerful StyleUnet along with a temporal-aware discriminator to achieve real-time realistic rendering. To enable robust animation for out-of-distribution poses, we propose a Motion Distribution Align module to compensate for the discrepancies between the training and testing motion distribution.Results and extensive experiments conducted in various experimental settings demonstrate the superiority of our proposed method, and a real-time live system is proposed to further push research into applications. The training and testing code will be released for research purposes.*

<img src="https://github.com/Xiang-Deng00/RAM-Avatar/blob/main/pipeline.png">


## Requirements
- python 3.9.17
- pytorch 2.0.0+cu118
- torchvision 0.15.1+cu118
- setuptools 68.0.0
- scikit-image 0.22.0
- numpy 1.25.2

## Datasets


1. Fit the Smpl-X parameters using ProxyCapV2. 
2. Fit the Faceverse parameters using Faceverse.
3. Render smpl and face maps using pytorch3d.
4. Construct the data directory as following.

dataset/train:

    |dataset/train
       |——keypoints_mmpose_hand
          |——00000001.json
          |——00000002.json
          |——...
       |——smpl_map
          |——00000001.png
          |——00000002.png
          |——...
       |——smpl_map_001
          |——00000001.png
          |——00000002.png
          |——...
       |——track2
          |——00000001.png
          |——00000002.png
          |——...
        |——00000001.png
        |——00000002.png
        |——...

## Train
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_train.py --from_json configs/train.json --name train --nump 4
```

## Test
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_test.py --from_json configs/test.json --name train --nump 4
```
## Acknowledgement 
This code is built upon [Styleavatar](https://github.com/LizhenWangT/StyleAvatar) and [CCNet](https://github.com/speedinghzl/CCNet). Thanks to the authors of these open source codes. 