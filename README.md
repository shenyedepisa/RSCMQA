# Copy-Move Forgery Detection and Question Answering for Remote Sensing Image

This is the initial version of the RS-CMQA dataset, RS-CMQA-B dataset, Real-RSCM dataset, RS-TQA dataset, RS-TQA-B dataset and Copy-Move Forgery Awareness Framework (CMFAF).

2024.9.5.	initial version

2024.12.15.	Updated code and dataset links

2024.5.24.	Updated code and dataset links

### Installation

##### python >=3.10

```
conda create -n tamper python=3.10
conda activate tamper
```

##### pytorch

[**install pytorch**](https://pytorch.org/)

```
# e.g. CUDA 11.8
# with conda
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# with pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

##### Install Packages

```
pip install -r requirements.txt
```

### Download Datasets

- ~~**Datasets-beta-1.0 is released at Baidu Drive** (2024.9.5). Available for download.~~

  Dataset-beta-1.0 only includes copy-move forgery

- ~~**Datasets-beta-1.1 is released at Baidu Drive**~ (2024.10.11).~~

  Dataset-beta-1.1 includes copy-move  and blurring tamper. For blurring tamper, the tampered region and the source region are treated as the same region

- **Dataset Release**

  the released datasets include RS-CMQA (RS-CMQA-B), RS-TQA (RS-TQA-B) and Real-RSCM

  Download link: [**Baidu Drive (224 x 224)** ](https://pan.baidu.com/s/1V-MIFmOnoEFRvFp3M8myJQ?pwd=CMQA) , **[Google Drive (224x224)](https://drive.google.com/drive/folders/1-1iMW6o_j1u16y5ZmckpqD2n1M9QVEIG?usp=drive_link)**

  Download link: [**Baidu Drive (512 x 512)** ](https://pan.baidu.com/s/1RG_EPz5rbMOcGvRkyS7wTQ?pwd=CMQA) 

- Dataset Directory: ` datasets/`

- Dataset Subdirectory: `datasets/subName/JsonFiles/`, `datasets/subName/JsonFilesBalanced/`, `datasets/subName/image/`, `datasets/subName/source/`, `datasets/subName/target/`, `datasets/subName/background/`, `datasets/subName/original/ (not used)`, 


### Download pre-trained weights

[**Download clip-b-32 weights from Hugging Face**](https://huggingface.co/openai/clip-vit-base-patch32/tree/main)

- Clip Directory: `models/clipModels/openai_clip_b_32/`

[**Download U-Net weights from Github**](https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale1.0_epoch2.pth) 

- U-Net Directory: `models/imageModels/milesial_UNet/`

### Start Training

```
python main.py
```

- Modify the experiment settings and hyperparameters in `src/config.py`

### Data Examples

![数据集](https://github.com/shenyedepisa/RSCMQA/blob/main/img/datasets.png)

### Citation

```
@article{zhang2024copymove,
    title={Copy-Move Forgery Detection and Question Answering for Remote Sensing Image}, 
    author={Z. Zhang and E. Zhao and Z. Wan and J. Nie and X. Liang and L. Huang},
    journal={arXiv preprint arXiv:2412.02575},
    year={2024},
}
```

### License

[**CC BY-NC-SA 4.0**](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en)

All images and their associated annotations in Global-TQA can be used for academic purposes only, **but any commercial use is prohibited.**
