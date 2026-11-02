
import torch
import numpy as np


def extract_features(model, loader, device):
    features = []
    pids = []
    camids = []

    model.eval()

    with torch.no_grad():
        for batch in loader:
            images = batch["img"].to(device)

            outputs = model(images)

            features.append(outputs.cpu())
            pids.extend(batch["pid"].cpu().numpy().tolist())
            camids.extend(batch["camid"].cpu().numpy().tolist())

    features = torch.cat(features, dim=0)

    return (
        features,
        np.asarray(pids),
        np.asarray(camids)
    )


def evaluate_market1501(
    model,
    query_loader,
    gallery_loader,
    device
):
    query_features, query_pids, query_camids = extract_features(
        model, query_loader, device
    )

    gallery_features, gallery_pids, gallery_camids = extract_features(
        model, gallery_loader, device
    )

    query_features = torch.nn.functional.normalize(
        query_features, p=2, dim=1
    )

    gallery_features = torch.nn.functional.normalize(
        gallery_features, p=2, dim=1
    )

    distance_matrix = 1 - torch.mm(
        query_features,
        gallery_features.t()
    )

    distance_matrix = distance_matrix.numpy()

    all_cmc = []
    all_ap = []

    max_rank = min(20, len(gallery_pids))

    for index in range(len(query_pids)):

        query_pid = query_pids[index]
        query_camid = query_camids[index]

        distances = distance_matrix[index]

        order = np.argsort(distances)

        matches = (
            gallery_pids[order] == query_pid
        ).astype(np.int32)

        remove = (
            (gallery_pids[order] == 0) |
            (gallery_pids[order] == -1) |
            (
                (gallery_pids[order] == query_pid) &
                (gallery_camids[order] == query_camid)
            )
        )

        keep = ~remove

        matches = matches[keep]

        if not np.any(matches):
            continue

        cumulative_matches = matches.cumsum()

        cmc = (
            cumulative_matches > 0
        ).astype(np.float32)

        cmc = cmc[:max_rank]

        if len(cmc) < max_rank:
            cmc = np.pad(
                cmc,
                (0, max_rank - len(cmc)),
                mode="edge"
            )

        all_cmc.append(cmc)

        precision = (
            cumulative_matches /
            (np.arange(len(matches)) + 1)
        )

        average_precision = (
            precision * matches
        ).sum() / matches.sum()

        all_ap.append(average_precision)

    if not all_cmc:
        raise RuntimeError(
            "No valid query identities found during evaluation."
        )

    cmc_scores = np.asarray(all_cmc).mean(axis=0)
    mean_ap = float(np.mean(all_ap))

    results = {
        "rank1": float(cmc_scores[0] * 100),
        "rank5": float(cmc_scores[4] * 100),
        "rank10": float(cmc_scores[9] * 100),
        "rank20": float(cmc_scores[19] * 100),
        "mAP": mean_ap * 100
    }

    return results
