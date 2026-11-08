
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def to_image(tensor):
    array = tensor.detach().cpu().float().permute(1, 2, 0).numpy()
    return np.clip(array * STD + MEAN, 0, 1)


def normalize_map(tensor):
    array = tensor.detach().cpu().float().squeeze().numpy()

    array = np.nan_to_num(array)
    array = array - array.min()

    if array.max() > 0:
        array = array / array.max()

    return array


def save_heatmap(original, heat, path, title):
    heat = np.asarray(
        Image.fromarray(
            (heat * 255).astype(np.uint8)
        ).resize(
            (original.shape[1], original.shape[0])
        ),
        dtype=np.float32
    ) / 255.0

    plt.figure(figsize=(5, 8))
    plt.imshow(original)
    plt.imshow(heat, cmap="jet", alpha=0.45)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def save_channel_weights(weights, path, title):
    values = weights.detach().cpu().float().flatten().numpy()

    plt.figure(figsize=(12, 4))
    plt.plot(np.arange(len(values)), values)
    plt.xlabel("Channel")
    plt.ylabel("Attention Weight")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def get_tensor(output):
    if torch.is_tensor(output):
        return output

    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                return item

    if isinstance(output, dict):
        for item in output.values():
            if torch.is_tensor(item):
                return item

    return None


@torch.no_grad()
def save_feature_visualizations(
    model,
    images,
    output_dir,
    epoch,
    max_images=4
):
    output_dir = (
        Path(output_dir)
        / "visualizations"
        / f"epoch_{epoch:03d}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    model = model.module if hasattr(model, "module") else model

    was_training = model.training
    model.eval()

    captured = {}
    handles = []

    def make_hook(name):
        def hook(module, inputs, output):
            tensor = get_tensor(output)

            if tensor is not None:
                captured[name] = tensor.detach()

        return hook

    # Capture feature maps from the main MSFANet stages.
    for name in ["stage1", "stage2", "stage3"]:
        if hasattr(model, name):
            module = getattr(model, name)

            handles.append(
                module.register_forward_hook(
                    make_hook(name)
                )
            )

    # Capture aggregation gate outputs.
    for name, module in model.named_modules():
        if ".gates." in name and name.endswith(".gate"):
            handles.append(
                module.register_forward_hook(
                    make_hook(name.replace(".", "_"))
                )
            )

    device = next(model.parameters()).device
    images = images[:max_images].to(device)

    model(images)

    for handle in handles:
        handle.remove()

    for image_index in range(images.size(0)):
        original = to_image(images[image_index])

        plt.imsave(
            output_dir / f"sample_{image_index + 1}_original.png",
            original
        )

        for map_name, batch_map in captured.items():

            if batch_map.ndim != 4:
                continue

            if image_index >= batch_map.size(0):
                continue

            sample_map = batch_map[image_index]

            if sample_map.shape[-2:] == (1, 1):
                save_channel_weights(
                    sample_map,
                    output_dir / (
                        f"sample_{image_index + 1}_{map_name}.png"
                    ),
                    map_name
                )

            else:
                # Mean absolute activation across channels.
                heat = sample_map.float().abs().mean(dim=0)

                heat = normalize_map(heat)

                save_heatmap(
                    original,
                    heat,
                    output_dir / (
                        f"sample_{image_index + 1}_{map_name}.png"
                    ),
                    map_name
                )

    if was_training:
        model.train()

    print(f"Visualizations saved to: {output_dir}")
