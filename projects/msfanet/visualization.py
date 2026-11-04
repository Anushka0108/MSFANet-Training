
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def _to_image(tensor):
    array = tensor.detach().cpu().float().permute(1, 2, 0).numpy()
    array = np.clip(array * STD + MEAN, 0, 1)
    return array


def _normalize_map(tensor):
    array = tensor.detach().cpu().float().squeeze().numpy()
    array = array - array.min()

    if array.max() > 0:
        array = array / array.max()

    return array


def _save_spatial_map(original, heat, path):
    heat = np.asarray(
        Image.fromarray((heat * 255).astype(np.uint8)).resize(
            (original.shape[1], original.shape[0])
        ),
        dtype=np.float32
    ) / 255.0

    plt.figure(figsize=(5, 8))
    plt.imshow(original)
    plt.imshow(heat, cmap="jet", alpha=0.45)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0)
    plt.close()


def _save_channel_weights(weights, path):
    values = weights.detach().cpu().float().flatten().numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(values)), values)
    plt.xlabel("Channel")
    plt.ylabel("Attention Weight")
    plt.title("Channel Attention Weights")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


@torch.no_grad()
def save_feature_visualizations(
    model, images, output_dir, epoch, max_images=4
):
    output_dir = (
        Path(output_dir) / "visualizations" / f"epoch_{epoch:03d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()

    device = next(model.parameters()).device
    model(images[:max_images].to(device))
    visual_maps = model.get_visual_maps()

    for image_index in range(min(max_images, images.size(0))):
        original = _to_image(images[image_index])

        Image.fromarray(
            (original * 255).astype(np.uint8)
        ).save(output_dir / f"sample_{image_index + 1}_original.png")

        for map_name, batch_map in visual_maps.items():
            if batch_map is None:
                continue

            sample_map = batch_map[image_index]

            if sample_map.shape[-2:] == (1, 1):
                _save_channel_weights(
                    sample_map,
                    output_dir / (
                        f"sample_{image_index + 1}_{map_name}_weights.png"
                    )
                )
            else:
                heat = _normalize_map(sample_map)

                _save_spatial_map(
                    original,
                    heat,
                    output_dir / (
                        f"sample_{image_index + 1}_{map_name}.png"
                    )
                )

    if was_training:
        model.train()
