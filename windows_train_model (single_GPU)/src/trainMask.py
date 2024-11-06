from tqdm import tqdm
import torch
import torch.nn.functional as F


def train_mask_model(
    _config, model, train_loader, trainLength, val_loader, valLength, device, logger
):
    bestLoss = 999999
    saveDir = _config["saveDir"]
    num_epochs = _config["thread_epoch"]
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-4,
        weight_decay=_config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=num_epochs, eta_min=1e-4
    )
    (
        trainLoss,
        trainMaeLoss,
        trainMseLoss,
        valLoss,
        valMaeLoss,
        valMseLoss,
    ) = ([], [], [], [], [], [])
    logger.info("Two steps, \nMaskModel Pre-train...")
    for epoch in range(num_epochs):
        maeLoss, mseLoss = 0, 0
        model.train()
        for i, data in tqdm(
            enumerate(train_loader, 0),
            total=len(train_loader),
            ncols=100,
            mininterval=1,
        ):
            question, answer, image, mask = data
            pred_mask = model(image.to(device))
            mae = F.l1_loss(mask.to(device), pred_mask)
            mse = F.mse_loss(mask.to(device), pred_mask)
            loss = mae + mse
            loss.backward()
            maeLoss += mae.cpu().item() * image.shape[0]
            mseLoss += mse.cpu().item() * image.shape[0]
            optimizer.step()
        trainMaeLoss.append(maeLoss / trainLength)
        trainMseLoss.append(mseLoss / trainLength)
        trainLoss.append(trainMaeLoss[epoch] + trainMseLoss[epoch])
        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"MaskModel Pre-train: epoch {epoch}, train loss: {trainLoss[epoch]:.5f}, "
            f"mae loss: {trainMaeLoss[epoch]:.5f}, mae loss: {trainMseLoss[epoch]:.5f},lr: {lr}"
        )
        scheduler.step()
        logger.info(f"Validation:")
        with torch.no_grad():
            model.eval()
            maeLoss, mseLoss = 0, 0
            for i, data in tqdm(
                enumerate(val_loader, 0),
                total=len(val_loader),
                ncols=100,
                mininterval=1,
            ):
                question, answer, image, type_str, mask, image_original = data
                pred_mask = model(image.to(device))
                mae = F.l1_loss(mask.to(device), pred_mask)
                mse = F.mse_loss(mask.to(device), pred_mask)
                maeLoss += mae.cpu().item() * image.shape[0]
                mseLoss += mse.cpu().item() * image.shape[0]
            valMaeLoss.append(maeLoss / valLength)
            valMseLoss.append(mseLoss / valLength)
            valLoss.append(valMseLoss[epoch] + valMaeLoss[epoch])
            logger.info(
                f"Epoch {epoch} , val loss: {valLoss[epoch]:.5f}, mae loss: {valMaeLoss[epoch]:.5f}, "
                f"mae loss: {valMseLoss[epoch]:.5f}"
            )
        if valLoss[epoch] < bestLoss:
            bestLoss = valLoss[epoch]
            torch.save(model, f"{saveDir}maskModel.pth")
