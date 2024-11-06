import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from src import Loader, SeqEncoder, ex, Logger
import torchvision.transforms as T
from tqdm import tqdm
import time
import torch.nn.functional as F
from models import CDModel
import copy


def test_model(_config, model, test_loader, test_length, device, logger, wandb_epoch=None, epoch=0):
    v1 = time.time()
    classes = _config["question_classes"]
    criterion = torch.nn.CrossEntropyLoss()
    logger.info(f"Testing:")
    with torch.no_grad():
        model.eval()
        accLoss, maeLoss, rmseLoss = 0, 0, 0

        countQuestionType = {str(i): 0 for i in range(1, classes + 1)}
        rightAnswerByQuestionType = {str(i): 0 for i in range(1, classes + 1)}

        for i, data in tqdm(
            enumerate(test_loader, 0),
            total=len(test_loader),
            ncols=100,
            mininterval=1,
        ):
            question, answer, image, type_str, mask, image_original = data
            pred, pred_mask = model(
                image.to(device), question.to(device), mask.to(device)
            )
            answer = answer.to(device)
            mae = F.l1_loss(mask.to(device), pred_mask)
            mse = F.mse_loss(mask.to(device), pred_mask)
            rmse = torch.sqrt(mse)
            # The ground truth of mask has not been normalized. (Which is intuitively weird)
            # This may be modified in future versions, but currently this method works better than directly normalizing the mask
            if not _config['normalize']:
                mae = mae / 255
                rmse = rmse / 255
            acc_loss = criterion(pred, answer)
            accLoss += acc_loss.cpu().item() * image.shape[0]
            maeLoss += mae.cpu().item() * image.shape[0]
            rmseLoss += rmse.cpu().item() * image.shape[0]
            answer = answer.cpu().numpy()
            pred = np.argmax(pred.cpu().detach().numpy(), axis=1)

            for j in range(answer.shape[0]):
                countQuestionType[type_str[j]] += 1
                if answer[j] == pred[j]:
                    rightAnswerByQuestionType[type_str[j]] += 1

        testAccLoss = accLoss / test_length
        testMaeLoss = maeLoss / test_length
        testRmseLoss = rmseLoss / test_length
        testLoss = testAccLoss + testRmseLoss + testMaeLoss
        logger.info(
            f"Epoch {epoch} , test loss: {testLoss:.5f}, acc loss : {testAccLoss:.5f}, "
            f"mae loss: {testMaeLoss:.5f}, rmse loss: {testRmseLoss:.5f}"
        )
        numQuestions = 0
        numRightQuestions = 0
        logger.info("Acc:")
        subclassAcc = {}
        accPerQuestionType = {str(i): [] for i in range(1, classes + 1)}
        for type_str in countQuestionType.keys():
            if countQuestionType[type_str] > 0:
                accPerQuestionType[type_str].append(
                    rightAnswerByQuestionType[type_str]
                    * 1.0
                    / countQuestionType[type_str]
                )
            else:
                accPerQuestionType[type_str].append(0)
            numQuestions += countQuestionType[type_str]
            numRightQuestions += rightAnswerByQuestionType[type_str]
            subclassAcc[type_str] = tuple(
                (countQuestionType[type_str], accPerQuestionType[type_str][0])
            )
        logger.info(
            "\t".join(
                [
                    f"{key}({subclassAcc[key][0]}) : {subclassAcc[key][1]:.5f}"
                    for key in subclassAcc.keys()
                ]
            )
        )

        # ave acc
        acc = numRightQuestions * 1.0 / numQuestions
        AA = 0
        for key in subclassAcc.keys():
            if wandb_epoch:
                wandb_epoch.log({"test " + key + " acc": subclassAcc[key][1]}, step=epoch)
            AA += subclassAcc[key][1]
        if _config['balance']:
            AA = AA / (len(subclassAcc) - 2)
        else:
            AA = AA / len(subclassAcc)
        v2 = time.time()
        logger.info(f"overall acc: {acc:.5f}\taverage acc: {AA:.5f}")
        if wandb_epoch:
            wandb_epoch.log(
                {
                    "test overall acc": acc,
                    "test average acc": AA,
                    "test loss": testLoss,
                    "test acc loss": testAccLoss,
                    "test mae loss": testMaeLoss,
                    "test rmse loss": testRmseLoss,
                    "test time cost": v2 - v1,
                },
                step=epoch,
            )


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    saveDir = _config["saveDir"]
    trainText = _config["trainText"]
    trainImg = _config["trainImg"]
    textHead = _config["textHead"]
    imageHead = _config["imageHead"]
    image_size = _config["imageSize"]
    Data = _config["DataConfig"]
    num_workers = _config["num_workers"]
    pin_memory = _config["pin_memory"]
    persistent_workers = _config["persistent_workers"]
    batch_size = _config["batch_size"]
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)
    log_file_name = (
        saveDir + "Test-" + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + ".log"
    )
    logger = Logger(log_file_name)
    source_img_size = _config["source_image_size"]
    seq_Encoder = SeqEncoder(_config, Data["allQuestionsJSON"], textTokenizer=textHead)
    # RGB
    IMAGENET_MEAN = [0.3833698, 0.39640951, 0.36896593]
    IMAGENET_STD = [0.21045856, 0.1946447, 0.18824594]
    data_transforms = {
        "image": T.Compose(
            [
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                T.Resize((image_size, image_size), antialias=True),
            ]
        ),
        "mask": T.Compose(
            [T.ToTensor(), T.Resize((image_size, image_size), antialias=True)]
        ),
    }
    print("Testing dataset preprocessing...")
    test_dataset = Loader(
        _config,
        Data["test"],
        seq_Encoder,
        source_img_size,
        textHead=textHead,
        imageHead=imageHead,
        train=False,
        transform=data_transforms,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weightsName = "weight_path"
    model = CDModel(
        _config,
        seq_Encoder.getVocab(question=True),
        seq_Encoder.getVocab(question=False),
        input_size=image_size,
        textHead=textHead,
        imageHead=imageHead,
        trainText=trainText,
        trainImg=trainImg,
    )
    state_dict = torch.load(weightsName,map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    test_length = len(test_dataset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=max(batch_size, 2),
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )
    test_model(_config, model, test_loader,test_length, device, logger)
