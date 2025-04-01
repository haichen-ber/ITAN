## 🌐 ITAN
Official implementation of the paper: Enhancing the Transferability of Adversarial Point Clouds by Initializing Transferable Adversarial Noise

## 🌟 Installation
Install the necessary dependencies following [AOF](https://github.com/code-roamer/AOF).

## 💾 Run
```bash
    NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=29506  baselines/attack_scripts/untarget_ITAN_attack.py --test_batch_size 35
    ```
## 📚 Acknowledgement
Many thanks to these excellent projects:
- [AOF](https://github.com/code-roamer/AOF)
- [PointNet/PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch), [DGCNN](https://github.com/WangYueFt/dgcnn), [PointConv](https://github.com/DylanWusee/pointconv_pytorch), [RS-CNN](https://github.com/Yochengliu/Relation-Shape-CNN)
- [Perturb/Add attack](https://github.com/xiangchong1/3d-adv-pc), [kNN attack](https://github.com/jinyier/ai_pointnet_attack), [Drop attack](https://github.com/tianzheng4/PointCloud-Saliency-Maps)
- [PU-Net](https://github.com/lyqun/PU-Net_pytorch), [DUP-Net](https://github.com/RyanHangZhou/DUP-Net)
- [ONet](https://github.com/autonomousvision/occupancy_networks), [ConvONet](https://github.com/autonomousvision/convolutional_occupancy_networks)
- [IF-Defense](https://github.com/Wuziyi616/IF-Defense)

## 📝 Citation

If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@article{chen2024enhancing,
  title={Enhancing the Transferability of Adversarial Point Clouds by Initializing Transferable Adversarial Noise},
  author={Chen, Hai and Zhao, Shu and Yan, Yuanting and Qian, Fulan},
  journal={IEEE Signal Processing Letters},
  volume={32},
  pages={201--205},
  year={2024},
  publisher={IEEE}
}
```
