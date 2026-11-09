
import os
import random
import csv
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

WEIGHT_DECAY = 5e-4

CHECKPOINT_PATH = Path(
    "/content/drive/MyDrive/MSFANet_Results/model.pth.tar-60"
)

OUTPUT_DIR = Path(
    "/content/drive/MyDrive/MSFANet_Results/MSFANet_Resume_60"
)

EXPECTED_CHECKPOINT_EPOCH = 60

MILESTONES = [150, 225, 300, 350]


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
    device = torch.device("cuda" if use_gpu else "cpu")

    print(f"Device: {device}")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError("Invalid checkpoint format.")

    required_keys = ["state_dict", "optimizer", "scheduler", "epoch"]

    missing_keys = [
        key for key in required_keys
        if key not in checkpoint
    ]

    if missing_keys:
        raise RuntimeError(
            f"Checkpoint is missing required keys: {missing_keys}"
        )

    checkpoint_epoch = int(checkpoint["epoch"])

    if checkpoint_epoch != EXPECTED_CHECKPOINT_EPOCH:
        raise RuntimeError(
            f"Expected checkpoint epoch {EXPECTED_CHECKPOINT_EPOCH}, "
            f"but found epoch {checkpoint_epoch}."
        )

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

    model.load_state_dict(
        checkpoint["state_dict"],
        strict=True
    )

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0003,
        weight_decay=WEIGHT_DECAY,
        amsgrad=True
    )

    optimizer.load_state_dict(checkpoint["optimizer"])

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=MILESTONES,
        gamma=0.1
    )

    scheduler.load_state_dict(checkpoint["scheduler"])

    start_epoch = checkpoint_epoch

    current_lr = optimizer.param_groups[0]["lr"]

    print("\nCheckpoint restored successfully")
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Completed epoch:", checkpoint_epoch)
    print("Next epoch:", start_epoch + 1)
    print("Final epoch:", EPOCHS)
    print("Learning rate:", current_lr)
    print("Scheduler milestones:", sorted(scheduler.milestones.elements()))
    print("Scheduler last epoch:", scheduler.last_epoch)

    complexity = calculate_complexity(
        model,
        HEIGHT,
        WIDTH
    )

    print("\nModel complexity:")
    for key, value in complexity.items():
        print(f"{key}: {value}")

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
        from openpyxl import Workbook, load_workbook

        workbook_path = OUTPUT_DIR / "msfanet_results.xlsx"

        if workbook_path.exists():
            workbook = load_workbook(workbook_path)
        else:
            workbook = Workbook()

        training_csv = OUTPUT_DIR / "training_batches.csv"

        if "Training Batches" in workbook.sheetnames:
            del workbook["Training Batches"]

        if training_csv.exists():
            training_sheet = workbook.create_sheet(
                "Training Batches"
            )

            with training_csv.open(newline="") as csv_file:
                reader = csv.reader(csv_file)

                for row in reader:
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

        print("\nExcel results saved:", workbook_path)

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
