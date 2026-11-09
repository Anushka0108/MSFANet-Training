
import os
import random
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import torchreid
from torchreid import models

from augmentations import build_train_transform
from complexity import calculate_complexity
from engine import MSFANetEngine
from visualization import save_feature_visualizations


SEED = 42
EPOCHS = 400
HEIGHT = 256
WIDTH = 128
BATCH_SIZE = 128
NUM_INSTANCES = 4
WORKERS = 4
INITIAL_LR = 0.065
WEIGHT_DECAY = 5e-4

CHECKPOINT_PATH = Path("projects/msfanet/model.pth.tar-190")
OUTPUT_DIR = Path("log/msfanet_market1501_resume_190")

START_EPOCH = 190
MILESTONES = [225, 300, 350]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(SEED)

    torch.set_num_threads(
        max(1, min(4, os.cpu_count() or 1))
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH.resolve()}"
        )

    use_gpu = torch.cuda.is_available()

    print(f"Device: {'CUDA' if use_gpu else 'CPU'}")

    datamanager = torchreid.data.ImageDataManager(
        root="data",
        sources="market1501",
        targets="market1501",
        height=HEIGHT,
        width=WIDTH,
        batch_size_train=BATCH_SIZE,
        batch_size_test=100,
        num_instances=NUM_INSTANCES,
        train_sampler="RandomIdentitySampler",
        workers=WORKERS,
        transforms=["random_flip"]
    )

    datamanager.train_loader.dataset.transform = (
        build_train_transform(HEIGHT, WIDTH)
    )

    model = models.build_model(
        name="msfanet",
        num_classes=datamanager.num_train_pids,
        loss="softmax",
        pretrained=False
    )

    if use_gpu:
        model = model.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=INITIAL_LR,
        weight_decay=WEIGHT_DECAY,
        amsgrad=True
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=MILESTONES,
        gamma=0.1
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False
    )

    model_state = checkpoint.get(
        "state_dict",
        checkpoint.get("model", checkpoint)
    )

    model.load_state_dict(
        model_state,
        strict=True
    )

    checkpoint_epoch = int(
        checkpoint.get("epoch", START_EPOCH)
    )

    if checkpoint_epoch != START_EPOCH:
        raise RuntimeError(
            f"Expected epoch {START_EPOCH}, "
            f"but checkpoint contains epoch {checkpoint_epoch}"
        )

    if "optimizer" not in checkpoint:
        raise RuntimeError(
            "This checkpoint does not contain optimizer state. "
            "Exact training resumption is not possible from this file."
        )

    optimizer.load_state_dict(checkpoint["optimizer"])

    if checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

        scheduler.milestones = Counter({
            milestone: 1 for milestone in MILESTONES
        })
    else:
        raise RuntimeError(
            "This checkpoint does not contain scheduler state. "
            "Cannot safely resume the original training schedule."
        )

    start_epoch = checkpoint_epoch

    print("\nCheckpoint restored successfully")
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Completed epoch:", start_epoch)
    print("Next epoch:", start_epoch + 1)
    print("Final epoch:", EPOCHS)
    print("Learning rate:", optimizer.param_groups[0]["lr"])
    print("Future milestones:", MILESTONES)

    complexity = calculate_complexity(
        model,
        HEIGHT,
        WIDTH
    )

    print("Model complexity:", complexity)

    engine = MSFANetEngine(
        datamanager=datamanager,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        weight_softmax=1.0,
        weight_triplet=0.5,
        triplet_margin=0.3,
        use_gpu=use_gpu,
        output_dir=str(OUTPUT_DIR)
    )

    engine.run(
        save_dir=str(OUTPUT_DIR),
        max_epoch=EPOCHS,
        start_epoch=start_epoch,
        print_freq=20,
        eval_freq=10,
        start_eval=1,
        ranks=[1, 5, 10, 20],
        normalize_feature=True
    )

    complexity_path = OUTPUT_DIR / "model_complexity.txt"

    with complexity_path.open("w") as handle:
        for key, value in complexity.items():
            handle.write(f"{key}: {value}\n")

    try:
        from openpyxl import load_workbook

        workbook_path = OUTPUT_DIR / "msfanet_results.xlsx"

        if workbook_path.exists():
            workbook = load_workbook(workbook_path)
        else:
            from openpyxl import Workbook
            workbook = Workbook()

        if "Training Batches" in workbook.sheetnames:
            del workbook["Training Batches"]

        training_csv = OUTPUT_DIR / "training_batches.csv"

        if training_csv.exists():
            training_sheet = workbook.create_sheet(
                "Training Batches"
            )

            with training_csv.open(newline="") as csv_file:
                for row in csv.reader(csv_file):
                    training_sheet.append(row)

        if "Model Complexity" in workbook.sheetnames:
            del workbook["Model Complexity"]

        complexity_sheet = workbook.create_sheet(
            "Model Complexity"
        )

        complexity_sheet.append(["Metric", "Value"])

        for key, value in complexity.items():
            complexity_sheet.append([key, value])

        if "Sheet" in workbook.sheetnames:
            del workbook["Sheet"]

        workbook.save(workbook_path)

    except Exception as error:
        print(f"Excel workbook update skipped: {error}")

    sample_batch = next(
        iter(datamanager.test_loader["market1501"]["query"])
    )

    save_feature_visualizations(
        model,
        sample_batch["img"],
        OUTPUT_DIR,
        epoch=EPOCHS,
        max_images=4
    )

    print("\nTraining continuation completed.")


if __name__ == "__main__":
    main()
