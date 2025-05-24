import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from src import Logger, train_mask_model
from transformers import optimization
from tqdm import tqdm
import time
import wandb
from test import test_model
import torch.nn.functional as F
from models import maskModel, CDModel


def train(_config, train_dataset, val_dataset, test_dataset, device, question_vocab):
    use_wandb = _config['use_wandb']
    wandb_step, wandb_epoch = None, None
    if use_wandb:
        wandb.login(key=_config["wandbKey"])
        wandb_epoch = None
        wandb_step = wandb.init(
            config=_config,
            project=_config["project"] + "_steps",
            name=_config["wandbName"],
            job_type=_config["job_type"],
            reinit=True,
        )
    start = time.time()
    textHead = _config["textHead"]
    imageHead = _config["imageHead"]
    trainText = _config["trainText"]
    trainImg = _config["trainImg"]
    image_size = _config["image_resize"]
    batch_size = _config["batch_size"]
    oneStep = _config["one_step"]
    opts = _config["opts"]
    num_epochs = _config["num_epochs"]
    learning_rate = _config["learning_rate"]
    saveDir = _config["saveDir"]
    miniStep = _config["steps"]
    need_mask = _config["need_mask"]
    is_scheduler = _config["scheduler"]
    num_workers = _config["num_workers"]
    pin_memory = _config["pin_memory"]
    persistent_workers = _config["persistent_workers"]
    classes = _config["question_classes"]
    if not os.path.isdir(saveDir):
        os.makedirs(saveDir)
    log_file_name = (
            saveDir + "log-" + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + ".log"
    )
    logger = Logger(log_file_name)
    logger.info(f"saveDir: {saveDir}")
    bestVal = 9999999
    bestAcc = 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=max(batch_size, 2),
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )

    optimizer_mask = None
    if oneStep:
        model = CDModel(
            _config,
            question_vocab.getVocab(),
            input_size=image_size,
            textHead=textHead,
            imageHead=imageHead,
            trainText=trainText,
            trainImg=trainImg,
        )
        if opts and need_mask:
            maskNet_params = [p for p in model.maskNet.parameters() if p.requires_grad]
            maskNet_ids = {id(p) for p in model.maskNet.parameters()}
            other_params = [
                p
                for p in model.parameters()
                if id(p) not in maskNet_ids
                if p.requires_grad
            ]
            optimizer_mask = torch.optim.Adam(maskNet_params, lr=1e-3)

            optimizer = torch.optim.Adam(
                other_params,
                lr=learning_rate,
                weight_decay=_config["weight_decay"],
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=learning_rate,
                weight_decay=_config["weight_decay"],
            )
            if _config["opt"] == "SGD":
                optimizer = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=learning_rate,
                    weight_decay=_config["weight_decay"],
                    momentum=0.9,
                )
    else:
        mask_model = maskModel(_config).to(device)
        train_mask_model(
            _config,
            mask_model,
            train_loader,
            len(train_dataset),
            val_loader,
            len(val_dataset),
            device,
            logger,
        )
        model = CDModel(
            _config,
            question_vocab.getVocab(),
            input_size=image_size,
            textHead=textHead,
            imageHead=imageHead,
            trainText=trainText,
            trainImg=trainImg,
        )
        if need_mask:
            for param in model.maskNet.parameters():
                param.requires_grad = False
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=_config["weight_decay"],
        )

    scheduler = None
    mask_scheduler = None
    if is_scheduler:
        lr_end = _config["end_learning_rate"]
        if _config["CosineAnnealingLR"]:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=num_epochs, eta_min=lr_end
            )
        elif _config["warmUp"]:
            scheduler = optimization.get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=5,
                num_training_steps=num_epochs,
                lr_end=lr_end,
                power=2,
            )
        if oneStep and opts:
            mask_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer_mask, T_max=num_epochs, eta_min=1e-4
            )

    criterion = torch.nn.CrossEntropyLoss()
    (
        trainLoss,
        trainAccLoss,
        trainMaeLoss,
        trainRmseLoss,
        trainKLLoss,
        valLoss,
        valAccLoss,
        valMaeLoss,
        valRmseLoss,
        valKLLoss,
        acc,
    ) = ([], [], [], [], [], [], [], [], [], [], [])

    accPerQuestionType = {str(i): [] for i in range(1, classes + 1)}
    logger.info(
        f"Started training... total epoch: {num_epochs}, batch-size: {int(batch_size * miniStep)}, step: {miniStep}"
    )
    called = False
    model.to(device)
    steps = 0
    for epoch in range(num_epochs):
        # train
        model.train()
        accLoss, maeLoss, rmseLoss, mae, mse, rmse, step_mae, step_rmse, klLoss, kl = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        logger.info(f"Epoch {epoch}")
        t1 = time.time()
        if oneStep and epoch >= _config["thread_epoch"] and need_mask and not called:
            called = True
            for param in model.maskNet.parameters():
                param.requires_grad = False
        for i, data in tqdm(
                enumerate(train_loader, 0),
                total=len(train_loader),
                ncols=100,
                mininterval=1,
        ):
            if need_mask:
                question, answer, image, mask = data
                pred, pred_mask, kl = model(
                    image.to(device), question.to(device))
                mae = F.l1_loss(mask.to(device), pred_mask)
                mse = F.mse_loss(mask.to(device), pred_mask)
                rmse = torch.sqrt(mse)
            else:
                question, answer, image = data
                pred, _, kl = model(
                    image.to(device), question.to(device), epoch=epoch
                )
            answer = answer.to(device)
            acc_loss = criterion(pred, answer)
            if need_mask:
                loss = 0.3 * rmse + 0.7 * acc_loss + kl
                # The ground truth of mask has not been normalized. (Which is intuitively weird)
                # This may be modified in future versions, but currently this method works better than directly normalizing the mask
                if not _config['normalize']:
                    mae = mae.cpu() / 255
                    rmse = rmse.cpu() / 255
                step_mae = mae.cpu().item()
                step_rmse = rmse.cpu().item()
            else:
                loss = acc_loss

            step_acc = acc_loss.cpu().item()

            if epoch == 0 and use_wandb:
                wandb_step.log(
                    {
                        "step loss": step_rmse + step_mae + step_acc,
                        "step acc loss": step_acc,
                        "step mae loss": step_mae,
                        "step rmse loss": step_rmse,
                        "step kl": kl
                    },
                    step=steps,
                )
            steps += 1
            accLoss += step_acc * image.shape[0]
            maeLoss += step_mae * image.shape[0]
            rmseLoss += step_rmse * image.shape[0]
            klLoss += kl * image.shape[0]
            # --------------------- L1 ---------------------------
            if _config["L1Reg"]:
                L1_reg = 0
                for param in model.parameters():
                    L1_reg += torch.sum(torch.abs(param))
                loss = (loss + L1_reg * 1e-7) / miniStep
            # -----------------------------------------------------
            else:
                loss = loss / miniStep
            loss.backward()
            if (i + 1) % miniStep == 0:
                if need_mask and oneStep and opts and epoch < _config["thread_epoch"]:
                    optimizer_mask.step()
                    optimizer_mask.zero_grad()
                optimizer.step()
                optimizer.zero_grad()

        trainAccLoss.append(accLoss / len(train_dataset))
        trainMaeLoss.append(maeLoss / len(train_dataset))
        trainRmseLoss.append(rmseLoss / len(train_dataset))
        trainKLLoss.append(klLoss / len(train_dataset))
        trainLoss.append(
            trainAccLoss[epoch] + trainRmseLoss[epoch] + trainMaeLoss[epoch] + trainKLLoss[epoch]
        )
        t2 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Training: epoch {epoch}, train loss: {trainLoss[epoch]:.5f}, acc loss : {trainAccLoss[epoch]:.5f}, "
            f"kl loss: {trainKLLoss[epoch]:.5f}"
            f"mae loss: {trainMaeLoss[epoch]:.5f}, rmse loss: {trainRmseLoss[epoch]:.5f},lr: {lr}\n"
        )
        if use_wandb:
            wandb_step.finish()

        if epoch == 0 and use_wandb:
            wandb_epoch = wandb.init(
                config=_config,
                project=_config["project"],
                name=_config["wandbName"],
                job_type=_config["job_type"],
                reinit=True,
            )
        if use_wandb:
            wandb_epoch.log(
                {
                    "train loss": trainLoss[epoch],
                    "train acc loss": trainAccLoss[epoch],
                    "train mae loss": trainMaeLoss[epoch],
                    "train rmse loss": trainRmseLoss[epoch],
                    "learning rate": lr,
                    "train time cost": t2 - t1,
                    "train kl loss": kl
                },
                step=epoch,
            )
        if is_scheduler:
            scheduler.step()
            if need_mask and oneStep and opts and epoch < _config["thread_epoch"]:
                mask_scheduler.step()

        # --------------------validation-------------------
        v1 = time.time()
        logger.info(f"Validation:")
        with torch.no_grad():
            model.eval()
            accLoss, maeLoss, rmseLoss, mae, mse, rmse, KLLoss = 0, 0, 0, 0, 0, 0, 0

            countQuestionType = {str(i): 0 for i in range(1, classes + 1)}
            rightAnswerByQuestionType = {str(i): 0 for i in range(1, classes + 1)}

            for i, data in tqdm(
                    enumerate(val_loader, 0),
                    total=len(val_loader),
                    ncols=100,
                    mininterval=1,
            ):
                if need_mask:
                    question, answer, image, type_str, mask, image_original = data
                    pred, pred_mask, kl = model(image.to(device), question.to(device))
                    mae = F.l1_loss(mask.to(device), pred_mask)
                    mse = F.mse_loss(mask.to(device), pred_mask)
                    rmse = torch.sqrt(mse)
                    # The ground truth of mask has not been normalized. (Which is intuitively weird)
                    # This may be modified in future versions, but currently this method works better than directly normalizing the mask
                    if not _config['normalize']:
                        mae = mae.cpu() / 255
                        rmse = rmse.cpu() / 255
                    mae = mae.cpu().item()
                    rmse = rmse.cpu().item()

                else:
                    question, answer, image, type_str, image_original = data
                    pred, _, kl = model(
                        image.to(device), question.to(device)
                    )
                answer = answer.to(device)

                acc_loss = criterion(pred, answer)
                accLoss += acc_loss.cpu().item() * image.shape[0]
                maeLoss += mae * image.shape[0]
                rmseLoss += rmse * image.shape[0]
                answer = answer.cpu().numpy()
                KLLoss += kl * image.shape[0]

                pred = np.argmax(pred.cpu().detach().numpy(), axis=1)

                for j in range(answer.shape[0]):
                    countQuestionType[type_str[j]] += 1
                    if answer[j] == pred[j]:
                        rightAnswerByQuestionType[type_str[j]] += 1

            valAccLoss.append(accLoss / len(val_dataset))
            valMaeLoss.append(maeLoss / len(val_dataset))
            valRmseLoss.append(rmseLoss / len(val_dataset))
            valKLLoss.append(KLLoss / len(val_dataset))
            valLoss.append(valAccLoss[epoch] + valRmseLoss[epoch] + valMaeLoss[epoch])
            if valLoss[epoch] < bestVal:
                bestVal = valLoss[epoch]
            logger.info(
                f"Epoch {epoch} , val loss: {valLoss[epoch]:.5f}, acc loss : {valAccLoss[epoch]:.5f}, "
                f"mae loss: {valMaeLoss[epoch]:.5f}, rmae loss: {valRmseLoss[epoch]:.5f}, "
                f"kl loss: {kl:.5f}"
            )

            numQuestions = 0
            numRightQuestions = 0
            logger.info("Acc:")
            subclassAcc = {}
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
                    (countQuestionType[type_str], accPerQuestionType[type_str][epoch])
                )
            logger.info(
                "\t".join(
                    [
                        f"{key}({subclassAcc[key][0]}) : {subclassAcc[key][1] * 100:.2f}"
                        for key in subclassAcc.keys()
                    ]
                )
            )

            # ave acc
            acc.append(numRightQuestions * 1.0 / numQuestions)
            if acc[epoch] > bestAcc:
                bestAcc = acc[epoch]
                torch.save(model.state_dict(), f"{saveDir}bestValAcc.pth")
            AA = 0
            for key in subclassAcc.keys():
                if use_wandb:
                    wandb_epoch.log(
                        {"val " + key + " acc": subclassAcc[key][1]}, step=epoch
                    )
                AA += subclassAcc[key][1]
            AA = AA / len(subclassAcc)

            v2 = time.time()
            logger.info(f"overall acc: {acc[epoch] * 100:.2f}\taverage acc: {AA * 100:.2f}")
            if use_wandb:
                wandb_epoch.log(
                    {
                        "val overall acc": acc[epoch],
                        "val average acc": AA,
                        "val loss": valLoss[epoch],
                        "val acc loss": valAccLoss[epoch],
                        "val mae loss": valMaeLoss[epoch],
                        "val rmse loss": valRmseLoss[epoch],
                        "validation time cost": v2 - v1,
                        "val kl loss": valKLLoss[epoch]
                    },
                    step=epoch,
                )
        torch.save(model.state_dict(), f"{saveDir}lastValModel.pth")
    test_model(
        _config,
        model,
        test_loader,
        len(test_dataset),
        device,
        logger,
        wandb_epoch,
        num_epochs,
    )
    end = time.time()
    logger.info(f"time used: {end - start} s")

    return model
