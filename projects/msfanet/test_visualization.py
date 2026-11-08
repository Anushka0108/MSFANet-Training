
import torch
import numpy as np
from pathlib import Path
from PIL import Image

import torchreid
from torchreid import metrics, models

from visualization import save_feature_visualizations


# ============================================================
# CONFIGURATION
# ============================================================

HEIGHT = 256
WIDTH = 128

BATCH_SIZE = 8
NUM_INSTANCES = 4

CHECKPOINT_PATH = "projects/msfanet/model.pth.tar-190"

OUTPUT_DIR = Path("log/msfanet_verification")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# VERIFICATION FUNCTION
# ============================================================

def check(condition, message):

    if not condition:
        raise RuntimeError("FAILED: " + message)

    print("[PASS]", message)


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(model):

    checkpoint_path = Path(CHECKPOINT_PATH)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path.resolve()}\n"
            "Update CHECKPOINT_PATH to the correct checkpoint location."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    state_dict = checkpoint["state_dict"]

    # Handle DataParallel checkpoints.
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict)

    print("\nCheckpoint loaded successfully")
    print("Checkpoint:", checkpoint_path)
    print("Epoch:", checkpoint.get("epoch", "Unknown"))
    print("Rank-1:", checkpoint.get("rank1", "Unknown"))

    return checkpoint


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(model, loader):

    all_features = []
    all_pids = []
    all_camids = []

    model.eval()

    with torch.no_grad():

        for data in loader:

            images = data["img"].to(DEVICE)

            features = model(images)

            all_features.append(features.cpu())

            all_pids.extend(data["pid"].tolist())
            all_camids.extend(data["camid"].tolist())

    return (
        torch.cat(all_features, dim=0),
        all_pids,
        all_camids
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n==========================================")
    print("       MSFANet VISUALIZATION TEST")
    print("==========================================")

    print("Device:", DEVICE)

    # --------------------------------------------------------
    # 1. LOAD MARKET-1501
    # --------------------------------------------------------

    print("\n[1] Loading Market-1501...")

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

    print(
        "Training identities:",
        datamanager.num_train_pids
    )

    # --------------------------------------------------------
    # 2. INITIALIZE MSFANET
    # --------------------------------------------------------

    print("\n[2] Initializing MSFANet...")

    model = models.build_model(
        name="msfanet",
        num_classes=datamanager.num_train_pids,
        loss="softmax",
        pretrained=False
    ).to(DEVICE)

    check(
        sum(p.numel() for p in model.parameters()) > 0,
        "MSFANet model initialized"
    )

    # --------------------------------------------------------
    # 3. LOAD TRAINED CHECKPOINT
    # --------------------------------------------------------

    print("\n[3] Loading trained checkpoint...")

    checkpoint = load_checkpoint(model)

    model.eval()

    check(
        all(torch.isfinite(p).all().item()
            for p in model.parameters()),
        "Model parameters contain no NaN or Inf"
    )

    # --------------------------------------------------------
    # 4. GET QUERY AND GALLERY LOADERS
    # --------------------------------------------------------

    print("\n[4] Loading query and gallery datasets...")

    query_loader = datamanager.test_loader[
        "market1501"
    ]["query"]

    gallery_loader = datamanager.test_loader[
        "market1501"
    ]["gallery"]

    check(
        len(query_loader.dataset) > 0,
        "Query images loaded"
    )

    check(
        len(gallery_loader.dataset) > 0,
        "Gallery images loaded"
    )

    print("Query images:", len(query_loader.dataset))
    print("Gallery images:", len(gallery_loader.dataset))

    # --------------------------------------------------------
    # 5. VERIFY FEATURE EXTRACTION
    # --------------------------------------------------------

    print("\n[5] Testing feature extraction...")

    query_batch = next(iter(query_loader))

    images = query_batch["img"][:4].to(DEVICE)

    check(
        images.shape[1:] == (3, HEIGHT, WIDTH),
        "Input image dimensions are correct"
    )

    with torch.no_grad():

        features = model(images)

    check(
        features.ndim == 2
        and features.shape[0] == images.size(0),
        "Feature extraction works"
    )

    check(
        torch.isfinite(features).all().item(),
        "Extracted features contain no NaN or Inf"
    )

    print("Feature shape:", tuple(features.shape))

    # --------------------------------------------------------
    # 6. GENERATE VISUALIZATIONS
    # --------------------------------------------------------

    print("\n[6] Generating feature visualizations...")

    save_feature_visualizations(
        model=model,
        images=query_batch["img"],
        output_dir=OUTPUT_DIR,
        epoch=checkpoint.get("epoch", 190),
        max_images=4
    )

    visualization_dir = (
        OUTPUT_DIR
        / "visualizations"
        / f"epoch_{int(checkpoint.get('epoch', 190)):03d}"
    )

    saved_images = list(
        visualization_dir.glob("*.png")
    )

    check(
        len(saved_images) > 0,
        "Visualization PNG files generated"
    )

    original_images = list(
        visualization_dir.glob("*original.png")
    )

    check(
        len(original_images) == 4,
        "Four original images saved"
    )

    print("\nGenerated visualization files:")

    for image_path in saved_images:

        with Image.open(image_path) as image:

            check(
                image.size == (WIDTH, HEIGHT)
                if "original" in image_path.name
                else True,
                image_path.name + " image checked"
            )

        print(" ", image_path.name)

    # --------------------------------------------------------
    # 7. EXTRACT QUERY FEATURES
    # --------------------------------------------------------

    print("\n[7] Extracting query features...")

    query_features, query_pids, query_camids = extract_features(
        model,
        query_loader
    )

    check(
        torch.isfinite(query_features).all().item(),
        "Query features are valid"
    )

    print("Query feature shape:", tuple(query_features.shape))

    # --------------------------------------------------------
    # 8. EXTRACT GALLERY FEATURES
    # --------------------------------------------------------

    print("\n[8] Extracting gallery features...")

    gallery_features, gallery_pids, gallery_camids = extract_features(
        model,
        gallery_loader
    )

    check(
        torch.isfinite(gallery_features).all().item(),
        "Gallery features are valid"
    )

    print(
        "Gallery feature shape:",
        tuple(gallery_features.shape)
    )

    # --------------------------------------------------------
    # 9. COMPUTE DISTANCE MATRIX
    # --------------------------------------------------------

    print("\n[9] Computing distance matrix...")

    distance_matrix = metrics.compute_distance_matrix(
        query_features,
        gallery_features,
        metric="euclidean"
    )

    check(
        torch.isfinite(distance_matrix).all().item(),
        "Distance matrix contains no NaN or Inf"
    )

    # --------------------------------------------------------
    # 10. RANK AND MAP EVALUATION
    # --------------------------------------------------------

    print("\n[10] Evaluating Rank-1 through Rank-20 and mAP...")

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

    # --------------------------------------------------------
    # 11. DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n==========================================")
    print("          MSFANet RESULTS")
    print("==========================================")

    print(f"Checkpoint Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"Rank-1:  {cmc[0] * 100:.2f}%")
    print(f"Rank-5:  {cmc[4] * 100:.2f}%")
    print(f"Rank-10: {cmc[9] * 100:.2f}%")
    print(f"Rank-20: {cmc[19] * 100:.2f}%")
    print(f"mAP:     {mean_ap * 100:.2f}%")

    print("\n==========================================")
    print("VISUALIZATION OUTPUT:")
    print(visualization_dir.resolve())

    print("\n==========================================")
    print("       ALL CHECKS COMPLETED")
    print("==========================================")


if __name__ == "__main__":
    main()
