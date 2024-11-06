from sacred import Experiment
import os

ex = Experiment("RSCD_1", save_git_info=False)


@ex.config
def config():
    # Wandb Config
    wandbName = "formal_two_step_ops"
    wandbKey = "116c9acc73067dd77655e21532d04392aff2174a"
    project = "Global_TQA"
    job_type = "train"

    balance = False
    normalize = False
    answer_number = 53
    if balance:
        answer_number = 53

    opts = True
    one_step = False
    all_epochs = 30
    num_epochs = all_epochs
    thread_epoch = num_epochs
    if not one_step:
        thread_epoch = int(all_epochs * 0.5)
        num_epochs = thread_epoch
        step_two_epoch = all_epochs - thread_epoch

    question_classes = 15
    learning_rate = 5e-5
    saveDir = "outputs/formal_two_step_ops/"
    new_data_path = "datasets/"
    source_image_size = 224
    image_resize = 224
    FUSION_IN = 768
    FUSION_HIDDEN = 512
    DROPOUT = 0.3
    resample = False
    pin_memory = True
    persistent_workers = True
    num_workers = 4

    learnable_mask = True
    img_only = False
    mask_only = False
    add_mask = True

    real_batch_size = 32
    batch_size = 32  # batch_size * steps == real_batch_size
    steps = int(real_batch_size / batch_size)
    weight_decay = 0
    opt = "Adam"
    scheduler = True
    CosineAnnealingLR = True
    warmUp = False
    L1Reg = False
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
    MAX_ANSWERS = 100
    LEN_QUESTION = 40
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

    imageHead = "clip_b_32_224"
    if imageHead == "clip_b_32_224":
        imageModelPath = "models/clipModels/openai_clip_b_32"
        imageSize = 224
        VISUAL_OUT = 768
    elif imageHead == "siglip-512":
        imageModelPath = "models/clipModels/siglip_512"
        imageSize = 512

    textHead = "clip_b_32_224"
    if textHead == "clip_b_32_224":
        textModelPath = "models/clipModels/openai_clip_b_32"
        QUESTION_OUT = 512
    elif textHead == "siglip-512":
        textModelPath = "models/clipModels/siglip_512"
        imageSize = 512

    attnConfig = {
        "embed_size": FUSION_IN,
        "heads": 6,
        "mlp_input": 768,
        "mlp_ratio": 4,
        "mlp_output": 768,
        "attn_dropout": 0.1,
    }
