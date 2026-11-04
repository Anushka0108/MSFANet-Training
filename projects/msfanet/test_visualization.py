
import sys
from pathlib import Path

import torch
import numpy as np

from PIL import Image

import torchreid
from torchreid import metrics, models

from visualization import save_feature_visualizations


HEIGHT = 256
WIDTH = 128
BATCH_SIZE = 8
NUM_INSTANCES = 4
OUTPUT_DIR = Path("log/msfanet_verification")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check(condition, message):
    if not condition:
        raise RuntimeError("FAILED: " + message)

    print("[PASS]", message)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n========== MSFANet Verification ==========")
    print("Device:", DEVICE)

    datamanager = torchreid.data.ImageDataManager(
        root="data",
        sources="market1501",
        targets="market1501",
        height=HEIGHT,
        width=WIDTH,
        batch_size_train=BATCH_SIZE,
        batch_size_test=8,
        num_instances=NUM_INSTANCES,
        train_sampler="RandomIdentitySampler",
        workers=0,
        transforms=["random_flip"]
    )

    check(
        datamanager.num_train_pids > 0,
        "Market-1501 training identities loaded"
    )

    model = models.build_model(
        name="msfanet",
        num_classes=datamanager.num_train_pids,
        loss="softmax",
        pretrained=False
    ).to(DEVICE)

    model.eval()

    check(
        sum(p.numel() for p in model.parameters()) > 0,
        "MSFANet model initialized"
    )

    query_loader = datamanager.test_loader["market1501"]["query"]
    gallery_loader = datamanager.test_loader["market1501"]["gallery"]

    query_batch = next(iter(query_loader))
    images = query_batch["img"][:4].to(DEVICE)

    check(
        images.shape[1:] == (3, HEIGHT, WIDTH),
        "Input image dimensions are correct"
    )

    with torch.no_grad():
        features = model(images)

    check(
        features.ndim == 2 and features.shape[0] == images.size(0),
        "Feature extraction works"
    )

    check(
        torch.isfinite(features).all().item(),
        "Extracted features contain no NaN or Inf"
    )

    visual_maps = model.get_visual_maps()

    check(
        len(visual_maps) > 0,
        "Visual maps are available"
    )

    print("\nVisual map verification:")

    for name, value in visual_maps.items():
        check(
            value is not None,
            name + " is available"
        )

        check(
            value.shape[0] == images.size(0),
            name + " has correct batch dimension"
        )

        check(
            torch.isfinite(value).all().item(),
            name + " contains no NaN or Inf"
        )

        print(
            f"  {name}: {tuple(value.shape)}"
        )

    check(
        any("branch" in name for name in visual_maps),
        "MSFA branch maps are captured"
    )

    check(
        any("gate" in name for name in visual_maps),
        "Aggregation gate maps are captured"
    )

    check(
        any("se" in name for name in visual_maps),
        "SE attention maps are captured"
    )

    check(
        "final_activation" in visual_maps,
        "Final activation map is captured"
    )

    print("\nChecking attention weight ranges:")

    for name, value in visual_maps.items():
        if "se" in name or "gate" in name:
            print(
                name,
                "min =", value.min().item(),
                "max =", value.max().item()
            )

    print("\nChecking visualization saving:")

    save_feature_visualizations(
        model,
        query_batch["img"],
        OUTPUT_DIR,
        epoch=0,
        max_images=4
    )

    visualization_dir = OUTPUT_DIR / "visualizations" / "epoch_000"

    saved_images = list(visualization_dir.glob("*.png"))

    check(
        len(saved_images) > 0,
        "Visualization PNG files were generated"
    )

    check(
        len(list(visualization_dir.glob("*original.png"))) == 4,
        "Four original images were saved"
    )

    for image_path in saved_images:
        with Image.open(image_path) as image:
            check(
                image.size == (WIDTH, HEIGHT),
                image_path.name + " has correct image dimensions"
            )

    print("\nChecking Rank and mAP evaluation:")

    def extract_features(loader):
        all_features = []
        all_pids = []
        all_camids = []

        for data in loader:
            imgs = data["img"].to(DEVICE)

            with torch.no_grad():
                feats = model(imgs)

            all_features.append(feats.cpu())
            all_pids.extend(data["pid"].tolist())
            all_camids.extend(data["camid"].tolist())

        return (
            torch.cat(all_features, dim=0),
            all_pids,
            all_camids
        )

    query_features, query_pids, query_camids = extract_features(
        query_loader
    )

    gallery_features, gallery_pids, gallery_camids = extract_features(
        gallery_loader
    )

    distance_matrix = metrics.compute_distance_matrix(
        query_features,
        gallery_features,
        metric="euclidean"
    )

    cmc, mean_ap = metrics.evaluate_rank(
        distance_matrix.numpy(),
        query_pids,
        gallery_pids,
        query_camids,
        gallery_camids,
        max_rank=20
    )

    check(
        len(cmc) >= 20,
        "CMC contains Rank-1 through Rank-20"
    )

    check(
        np.isfinite(cmc[:20]).all(),
        "CMC values are finite"
    )

    check(
        np.isfinite(mean_ap),
        "mAP is finite"
    )

    check(
        np.all(np.diff(cmc[:20]) >= -1e-7),
        "CMC values are non-decreasing"
    )

    print("\nEvaluation results (percentages):")
    print(f"Rank-1:  {cmc[0] * 100:.2f}%")
    print(f"Rank-5:  {cmc[4] * 100:.2f}%")
    print(f"Rank-10: {cmc[9] * 100:.2f}%")
    print(f"Rank-20: {cmc[19] * 100:.2f}%")
    print(f"mAP:     {mean_ap * 100:.2f}%")

    print("\nVisualization output:", visualization_dir.resolve())
    print("\n========== ALL CHECKS COMPLETED ==========")


if __name__ == "__main__":
    main()
