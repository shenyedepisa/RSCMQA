# RS-CMQA: Model and dataset for global remote sensing image copy-move forgery question answering

This is the initial version of the RS-CMQA dataset, RS-CMQA-B dataset and Copy-Move Forgery Awareness Framework (CMFAF). 

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

- **Datasets V1.0 is released at Baidu Drive** (2024.9.5)

- **Datasets V2.0 is released at Baidu Drive** (2024.10.11)

  

- Dataset Directory: ` datasets/`

- Dataset Subdirectory: `datasets/JsonFiles/`, `datasets/JsonFilesBalanced/`, `datasets/image/`, `datasets/source/`, `datasets/target/`, `datasets/background/`


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

![数据集](C:\Users\pizza\Desktop\开题\素材\数据集.png)



### Citation

```
@unpublished{global2025aaai,
  title = {Global-TQA: Global Remote Sensing Image Tampering Question and Answering},
  url={https://anonymous.4open.science/r/TQA},
  month = {8},
  year = {2024}
}
```

### License

[**CC BY-NC-SA 4.0**](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en)

All images and their associated annotations in Global-TQA can be used for academic purposes only, **but any commercial use is prohibited.**
