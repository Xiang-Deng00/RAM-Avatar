<div align="center">

# **RAM-Avatar: Real-Time Photo-Realistic Full-Body Avatar from Monocular Video**  
*(CVPR 2024)*

**Xiang Deng<sup>1</sup>**, [Zerong Zheng](https://zhengzerong.github.io/)<sup>2</sup>,  
[Yuxiang Zhang](https://zhangyux15.github.io/)<sup>1</sup>, [Jingxiang Sun](https://mrtornado24.github.io/)<sup>1</sup>,  
Chao Xu<sup>2</sup>, XiaoDong Yang<sup>3</sup>, [Lizhen Wang](https://lizhenwangt.github.io/)<sup>1</sup>, [Yebin Liu](https://www.liuyebin.com)<sup>1</sup>

<sup>1</sup>Tsinghua University <sup>2</sup>NNKosmos Technology <sup>3</sup>Li Auto

---

📄 [Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Deng_RAM-Avatar_Real-time_Photo-Realistic_Avatar_from_Monocular_Videos_with_Full-body_Control_CVPR_2024_paper.pdf) &nbsp;|&nbsp; 🎥 [Video Demo] &nbsp;|&nbsp; 

</div>

---

## 🌟 Overview

We present **RAM-Avatar**, a novel approach for learning **real-time, photo-realistic full-body avatars** from monocular videos. Our method enables **full-body control** with high-fidelity rendering of facial expressions, hand gestures, and body textures — all while maintaining real-time performance.

<div align="center">
  <img src="https://github.com/Xiang-Deng00/RAM-Avatar/blob/main/sample_results.png" width="800" alt="Sample Results"/>
</div>

## 📚 Abstract

This paper advances the practicality of human avatar learning by introducing **RAM-Avatar**, a framework that learns real-time, photo-realistic avatars from monocular videos with full-body controllability. To model fine-grained variations in facial expressions and hand gestures, we employ dedicated statistical templates. For the body, a sparsely computed dual attention mechanism enhances texture fidelity on torso and limbs. Built upon this, a lightweight yet powerful **StyleUNet** architecture, coupled with a temporal-aware discriminator, enables efficient and realistic rendering at real-time speeds. To ensure robust animation under out-of-distribution poses, we propose a **Motion Distribution Alignment (MDA)** module that reduces domain shift between training and inference. Extensive experiments validate the superiority of our method in both qualitative and quantitative evaluations. We further demonstrate its practical potential via a real-time live avatar system. Code and models will be released for research purposes.

<div align="center">
  <img src="https://github.com/Xiang-Deng00/RAM-Avatar/blob/main/pipeline.png" width="800" alt="Method Pipeline"/>
</div>

---

## ⚙️ Requirements

```bash
- Python 3.9.17
- PyTorch 2.0.0+cu118
- TorchVision 0.15.1+cu118
- setuptools 68.0.0
- scikit-image 0.22.0
- numpy 1.25.2
```

> 💡 We recommend using a Conda environment:
> ```bash
> conda create -n ramavatar python=3.9
> conda activate ramavatar
> pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118
> pip install scikit-image numpy setuptools
> ```

---

## 🗂️ Dataset Preparation

To train RAM-Avatar, prepare your dataset as follows:

1. **Estimate SMPL-X parameters** using [ProxyCapV2](https://github.com/eth-siplab/ProxyCap).
2. **Fit FaceVerse parameters** for facial dynamics.
3. **Render SMPL and facial maps** using [PyTorch3D](https://github.com/facebookresearch/pytorch3d).
4. Organize the data directory structure:

```
dataset/train/
├── keypoints_mmpose_hand/
│   ├── 00000001.json
│   ├── 00000002.json
│   └── ...
├── smpl_map/
│   ├── 00000001.png
│   ├── 00000002.png
│   └── ...
├── smpl_map_001/
│   ├── 00000001.png
│   ├── 00000002.png
│   └── ...
├── track2/
│   ├── 00000001.png
│   ├── 00000002.png
│   └── ...
├── 00000001.png          # Original frames
├── 00000002.png
└── ...
```

### 📦 Pretrained Checkpoints & Datasets

You can download our pre-trained models and sample datasets here:

- 🔗 **[Pretrained Checkpoints](https://pan.baidu.com/s/10l3gb6JEADHZBRMBMpAZjA?pwd=i8gt)** (Password: `i8gt`)  
- 🔗 **[Sample Dataset](https://pan.baidu.com/s/1aSEAPZWV62Pc8CDvEZh1JA?pwd=e4wp)** (Password: `e4wp`)  
*— Shared via Baidu Wangpan Super VIP*

> ⚠️ Note: These links are hosted on Baidu Netdisk. International users may need a download accelerator.

---

## 🏃 Training

Single GPU:
```bash
CUDA_VISIBLE_DEVICES=0 python main_train.py --from_json configs/train.json --name train --nump 0
```

Multi-GPU (4 GPUs):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_train.py --from_json configs/train.json --name train --nump 4
```

---

## 🧪 Testing

```bash
CUDA_VISIBLE_DEVICES=0 python main_test.py --from_json configs/test.json --name train --nump 0
```

---

## 🙏 Acknowledgements

This work is built upon the following excellent open-source projects. We thank the authors for their contributions:

- [StyleAvatar](https://github.com/LizhenWangT/StyleAvatar)
- [CCNet](https://github.com/speedinghzl/CCNet)

---

## 📎 Citation

If you find our work useful in your research, please cite:

```bibtex
@inproceedings{deng2024ram,
  title     = {RAM-Avatar: Real-Time Photo-Realistic Avatar from Monocular Videos with Full-Body Control},
  author    = {Deng, Xiang and Zheng, Zerong and Zhang, Yuxiang and Sun, Jingxiang and Xu, Chao and Yang, Xiaodong and Wang, Lizhen and Liu, Yebin},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {1996--2007},
  year      = {2024}
}
```

---

<div align="center">
  <small>© 2025 RAM-Avatar Authors. This project is for academic purposes only.</small>
</div>
