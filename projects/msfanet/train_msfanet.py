
import os
import sys
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torchreid

from projects.msfanet.config import *
from projects.msfanet.excel_logger import ExcelLogger
from projects.msfanet.evaluation import evaluate_market1501
from projects.msfanet.complexity import calculate_complexity


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path, model, optimizer, epoch, best_map):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_mAP": best_map
    }, path)


def main():

    set_seed(RANDOM_SEED)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not found. Run this training script in Google Colab "
            "with a GPU runtime."
        )

    device = torch.device("cuda")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = OUTPUT_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("Device:", device)
    print("Dataset root:", DATASET_ROOT)
    print("Output directory:", OUTPUT_DIR)

    datamanager = torchreid.data.ImageDataManager(
        root=str(DATASET_ROOT),
        sources=DATASET_NAME,
        targets=DATASET_NAME,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        batch_size_train=BATCH_SIZE,
        batch_size_test=128,
        num_instances=NUM_INSTANCES,
        train_sampler="RandomIdentitySampler",
        workers=NUM_WORKERS,
        use_gpu=True,
        transforms=[
            "random_flip",
            "random_crop",
            "random_erase"
        ]
    )

    model = torchreid.models.build_model(
        name=MODEL_NAME,
        num_classes=datamanager.num_train_pids,
        loss="triplet",
        pretrained=False,
        use_gpu=True
    )

    model = model.to(device)

    classification_loss = nn.CrossEntropyLoss()

    triplet_loss = torchreid.losses.TripletLoss(
        margin=TRIPLET_MARGIN
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        amsgrad=AMSGRAD
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=LR_MILESTONES,
        gamma=LR_DECAY
    )

    logger = ExcelLogger(str(OUTPUT_DIR))

    complexity = calculate_complexity(
        model,
        IMAGE_HEIGHT,
        IMAGE_WIDTH
    )

    config = {
        "Model": MODEL_NAME,
        "Dataset": DATASET_NAME,
        "Epochs": EPOCHS,
        "Batch Size": BATCH_SIZE,
        "Images per Identity": NUM_INSTANCES,
        "Image Height": IMAGE_HEIGHT,
        "Image Width": IMAGE_WIDTH,
        "Optimizer": OPTIMIZER,
        "AMSGrad": AMSGRAD,
        "Initial Learning Rate": LEARNING_RATE,
        "Weight Decay": WEIGHT_DECAY,
        "Classification Loss Weight": CLASSIFICATION_WEIGHT,
        "Triplet Loss Weight": TRIPLET_WEIGHT,
        "Triplet Margin": TRIPLET_MARGIN,
        "Pretrained": False,
        "Random Seed": RANDOM_SEED,
        "LR Milestones": str(LR_MILESTONES),
        "LR Decay": LR_DECAY,
        "Device": str(device)
    }

    best_map = 0.0
    best_epoch = 0

    train_loader = datamanager.train_loader

    test_loaders = datamanager.test_loader[DATASET_NAME]

    query_loader = test_loaders["query"]
    gallery_loader = test_loaders["gallery"]

    print("Training images:", len(train_loader))
    print("Number of classes:", datamanager.num_train_pids)
    print("Total parameters:", complexity["Total Parameters (M)"], "M")

    for epoch in range(1, EPOCHS + 1):

        model.train()

        total_loss_sum = 0.0
        classification_loss_sum = 0.0
        triplet_loss_sum = 0.0
        correct = 0
        total = 0
        batch_count = 0

        current_lr = optimizer.param_groups[0]["lr"]

        print("\nEpoch [{}/{}]".format(epoch, EPOCHS))

        for batch_index, data in enumerate(train_loader):

            images = data["img"].to(device)
            pids = data["pid"].to(device)

            optimizer.zero_grad()

            logits, features = model(images)

            loss_x = classification_loss(logits, pids)
            loss_t = triplet_loss(features, pids)

            loss = (
                CLASSIFICATION_WEIGHT * loss_x
                + TRIPLET_WEIGHT * loss_t
            )

            loss.backward()
            optimizer.step()

            predictions = torch.argmax(logits, dim=1)

            correct += (predictions == pids).sum().item()
            total += pids.size(0)

            total_loss_sum += loss.item()
            classification_loss_sum += loss_x.item()
            triplet_loss_sum += loss_t.item()

            batch_count += 1

            if (batch_index + 1) % PRINT_FREQUENCY == 0:
                print(
                    "Batch [{}/{}] Loss: {:.4f}".format(
                        batch_index + 1,
                        len(train_loader),
                        loss.item()
                    )
                )

        average_loss = total_loss_sum / max(batch_count, 1)
        average_loss_x = classification_loss_sum / max(batch_count, 1)
        average_loss_t = triplet_loss_sum / max(batch_count, 1)

        accuracy = 100.0 * correct / max(total, 1)

        print("Classification Loss:", average_loss_x)
        print("Triplet Loss:", average_loss_t)
        print("Total Loss:", average_loss)
        print("Training Accuracy:", accuracy)

        logger.add_training(
            epoch,
            average_loss_x,
            average_loss_t,
            average_loss,
            accuracy,
            current_lr
        )

        if epoch % EVALUATE_EVERY == 0:

            print("Evaluating Market-1501...")

            results = evaluate_market1501(
                model,
                query_loader,
                gallery_loader,
                device
            )

            print(
                "Rank-1: {:.2f}% | Rank-5: {:.2f}% | "
                "Rank-10: {:.2f}% | Rank-20: {:.2f}% | "
                "mAP: {:.2f}%".format(
                    results["rank1"],
                    results["rank5"],
                    results["rank10"],
                    results["rank20"],
                    results["mAP"]
                )
            )

            logger.add_evaluation(
                epoch,
                results["rank1"],
                results["rank5"],
                results["rank10"],
                results["rank20"],
                results["mAP"]
            )

            if results["mAP"] > best_map:

                best_map = results["mAP"]
                best_epoch = epoch

                save_checkpoint(
                    checkpoint_dir / "best_model.pth",
                    model,
                    optimizer,
                    epoch,
                    best_map
                )

                print("Best model saved.")

        scheduler.step()

        save_checkpoint(
            checkpoint_dir / "latest_model.pth",
            model,
            optimizer,
            epoch,
            best_map
        )

        logger.save(config, complexity)

    save_checkpoint(
        checkpoint_dir / "final_model.pth",
        model,
        optimizer,
        EPOCHS,
        best_map
    )

    config["Best Epoch"] = best_epoch
    config["Best mAP (%)"] = best_map

    logger.save(config, complexity)

    print("\nTraining completed.")
    print("Best epoch:", best_epoch)
    print("Best mAP:", best_map)
    print("Results:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
