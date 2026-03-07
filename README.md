<div align="center">
<h1>[CVPR'26] Learned Image Compression via Sparse Attention and Adaptive Frequency</h1>
<h3>Huidong Ma, Xinyan Shi, Hui Sun, Xiaofei Yue, Xiaoguang Liu, Gang Wang, Wentong Cai</h3>
</div>

# Running
### Evaluate
```
python eval.py --checkpoint <ckpt_path> --data <data_dir1> <data_dir2> --cuda --gpu <gpu_id>
```
For example: 
```
python eval.py --checkpoint mse_0.05.pth.tar --data ./Kodak ./CLIC ./Tecnick --cuda --gpu 0
```

# Results
Our models are trained on the [OpenImages](https://github.com/openimages) with MSE as the optimization target. The pre-trained models are available at [Link](https://drive.google.com/drive/folders/1TZlDDxYhMyRKiQeCbDgtr6W-S-lm7tiz?usp=drive_link).  
The RD results across the [Kodak](https://r0k.us/graphics/kodak/), [CLIC](https://storage.googleapis.com/clic_datasets/clic2020_professional_valid.zip), and [Tecnick](https://master.dl.sourceforge.net/project/testimages/OLD/OLD_SAMPLING/testimages.zip?viasf=1) datasets are as follows:
```
{
    "Kodak": {
        "PSNR": [38.08461, 36.13439, 34.27731, 32.47287, 30.63940, 29.14685],
        "bpp":  [0.83036, 0.58840, 0.41229, 0.27708, 0.17135, 0.10424]
    },
    "CLIC": {
        "PSNR": [38.79493, 37.14013, 35.55986, 34.01956, 32.40954, 31.09386],
        "bpp":  [0.60765, 0.41692, 0.29317, 0.20046, 0.12783, 0.07999]
    },
    "Tecnick": {
        "PSNR": [38.83482, 37.28899, 35.78379, 34.24225, 32.66148, 31.32328],
        "bpp":  [0.56136, 0.38974, 0.27949, 0.19800, 0.13328, 0.08987]
    }
}
```

<!--
# Citation
If you are interested in our work, we hope you might consider starring our repository and citing our paper:
```
```
-->

# Acknowledgment
The code is based on [CompressAI](https://github.com/InterDigitalInc/CompressAI), [DCAE](https://github.com/CVL-UESTC/DCAE), and [AuxT](https://github.com/qingshi9974/AuxT). Thanks for these great works.

# Contact
Email: mahd@nbjl.nankai.edu.cn  
Nankai-Baidu Joint Laboratory (NBJL)
