from sacred import Experiment
import os

ex = Experiment("TGRS_RSCMQA", save_git_info=False)


@ex.config
def config():
    # save
    use_wandb = False
    subDir = "real_demo"
    # Wandb Config
    # https://docs.wandb.ai/quickstart/
    wandbName = subDir
    wandbKey = ""
    project = "TGRS_RSCMQA"
    job_type = "train"

    balance = False
    need_mask = True
    normalize = False
    opts = True
    one_step = True
    num_epochs = 30
    thread_epoch = 20
    if balance == False:
        num_epochs = 20
        thread_epoch = 13

    learning_rate = 5e-5  # 5e-5
    num_workers = 2
    # datasetName = 'RS_TQA_224'
    # datasetName = 'RS_CMQA_224'
    datasetName = 'Real_RSCM'
    outputs = "./outputs/"
    saveDir = os.path.join(outputs, subDir + '/')
    new_data_path = f'./datasets/{datasetName}/'

    answer_number = 53
    question_classes = 15
    if datasetName == 'RSCMQA_224':
        answer_number = 51
        question_classes = 14
    if datasetName == 'CM_dataset':
        answer_number = 50
        question_classes = 14
        num_epochs = 30
        thread_epoch = 20

    real_batch_size = 32
    batch_size = 32  # batch_size * steps == real_batch_size
    steps = int(real_batch_size / batch_size)
    weight_decay = 0
    opt = "Adam"
    scheduler = True
    CosineAnnealingLR = True

    source_image_size = 512
    image_resize = 224

    FUSION_IN = 768
    FUSION_HIDDEN = 512
    DROPOUT = 0.3
    imageHead = "clip_b_32_224"
    textHead = "clip_b_32_224"

    add_mask = True
    pin_memory = True
    persistent_workers = True
    seg = False
    warmUp = False
    L1Reg = False
    resample = False
    trainText = True
    trainImg = True
    finetuneMask = True

    if scheduler:
        end_learning_rate = 1e-6

    json_path = os.path.join(new_data_path, 'JsonFiles')
    if balance:
        json_path = os.path.join(new_data_path, 'JsonFilesBalanced')
    DataConfig = {
        "images_path": os.path.join(new_data_path, "image"),
        "sourceMask_path": os.path.join(new_data_path, "source"),
        "targetMask_path": os.path.join(new_data_path, "target"),
        "backgroundMask_path": os.path.join(new_data_path, "background"),
        "seg_path": os.path.join(new_data_path, "segmentation"),
        "answersJson": os.path.join(json_path, "Answers.json"),
        "allQuestionsJSON": os.path.join(json_path, "All_Questions.json"),
        "train": {
            "imagesJSON": os.path.join(json_path, "All_Images.json"),
            "questionsJSON": os.path.join(json_path, "Train_Questions.json"),
        },
        "val": {
            "imagesJSON": os.path.join(json_path, "All_Images.json"),
            "questionsJSON": os.path.join(json_path, "Val_Questions.json"),
        },
        "test": {
            "imagesJSON": os.path.join(json_path, "All_Images.json"),
            "questionsJSON": os.path.join(json_path, "Test_Questions.json"),
        },
    }
    LEN_QUESTION = 30
    clipList = [
        "clip",
        "rsicd",
        "clip_b_32_224",
        "clip_b_16_224",
        "clip_l_14_224",
        "clip_l_14_336",
    ]
    vitList = ["vit-b", "vit-s", "vit-t"]
    maskHead = "unet"
    if maskHead == "unet":
        maskModelPath = (
            "models/imageModels/milesial_UNet/unet_carvana_scale1.0_epoch2.pth"
        )

    if imageHead == "clip_b_32_224":
        imageModelPath = "models/clipModels/openai_clip_b_32"
        VISUAL_OUT = 768
        image_resize = 224
    elif imageHead == "siglip_512":
        imageModelPath = "models/clipModels/siglip_512"
        image_resize = 512
    else:
        # vgg, resnet
        image_resize = 256
        imageModelPath = ""
    if textHead == "clip_b_32_224":
        textModelPath = "models/clipModels/openai_clip_b_32"
        QUESTION_OUT = 512
    elif textHead == "siglip_512":
        textModelPath = "models/clipModels/siglip_512"
        QUESTION_OUT = 768
    elif textHead == "skipthoughts":
        textModelPath = "models/textModels/skip-thoughts"
        QUESTION_OUT = 2400
    else:
        textModelPath = ""
        QUESTION_OUT = 2400
    attnConfig = {
        "embed_size": FUSION_IN,
        "heads": 6,
        "mlp_input": 768,
        "mlp_ratio": 4,
        "mlp_output": 768,
        "attn_dropout": 0.1,
    }
