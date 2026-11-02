
import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torchreid


def normalize_map(feature_map):
    feature_map = np.maximum(feature_map, 0)

    minimum = feature_map.min()
    maximum = feature_map.max()

    if maximum > minimum:
        feature_map = (
            (feature_map - minimum) /
            (maximum - minimum)
        )

    return feature_map


def save_feature_visualizations(name, activation, output_dir):

    folder = output_dir / name.replace(".", "_")
    folder.mkdir(parents=True, exist_ok=True)

    feature_map = activation[0].float().numpy()

    mean_map = feature_map.mean(axis=0)
    mean_map = normalize_map(mean_map)

    plt.figure(figsize=(7, 6))
    plt.imshow(mean_map, cmap="jet")
    plt.axis("off")
    plt.title(name + " Mean Activation")
    plt.savefig(
        folder / "mean_activation.png",
        bbox_inches="tight"
    )
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(feature_map.flatten(), bins=50)
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency")
    plt.title(name + " Activation Distribution")
    plt.savefig(
        folder / "activation_histogram.png",
        bbox_inches="tight"
    )
    plt.close()

    selected = feature_map[:min(16, feature_map.shape[0])]

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    for index, axis in enumerate(axes.flat):

        axis.axis("off")

        if index < len(selected):
            axis.imshow(
                normalize_map(selected[index]),
                cmap="viridis"
            )

    fig.suptitle(name + " Channel Activations")

    fig.savefig(
        folder / "channel_activations.png",
        bbox_inches="tight"
    )

    plt.close(fig)


def save_attention_visualization(name, attention, output_dir):

    folder = output_dir / "attention_weights"
    folder.mkdir(parents=True, exist_ok=True)

    weights = attention[0].flatten().numpy()

    plt.figure(figsize=(12, 4))
    plt.bar(np.arange(len(weights)), weights)

    plt.xlabel("Channel")
    plt.ylabel("Attention Weight")
    plt.title(name + " SE Attention Weights")

    plt.savefig(
        folder / (name.replace(".", "_") + ".png"),
        bbox_inches="tight"
    )

    plt.close()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True
    )

    parser.add_argument(
        "--image",
        required=True
    )

    parser.add_argument(
        "--output",
        default="activation_maps"
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = torchreid.models.build_model(
        name="msfanet",
        num_classes=751,
        loss="triplet",
        pretrained=False,
        use_gpu=torch.cuda.is_available()
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint
    )

    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    image_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(args.image).convert("RGB")
    input_tensor = image_transform(image)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    activations = {}
    attention_weights = {}
    hooks = []

    def activation_hook(name):

        def hook(module, inputs, output):

            if isinstance(output, torch.Tensor):
                activations[name] = output.detach().cpu()

        return hook

    def attention_hook(name):

        def hook(module, inputs, output):

            if module.last_attention is not None:
                attention_weights[name] = (
                    module.last_attention.cpu()
                )

        return hook

    from torchreid.models.msfanet.se_block import SEBlock

    for name, module in model.named_modules():

        if name in [
            "stem",
            "stage1",
            "stage2",
            "stage3"
        ]:
            hooks.append(
                module.register_forward_hook(
                    activation_hook(name)
                )
            )

        elif any(
            branch in name
            for branch in [
                "branch1",
                "branch2",
                "branch3"
            ]
        ):
            hooks.append(
                module.register_forward_hook(
                    activation_hook(name)
                )
            )

        if isinstance(module, SEBlock):
            hooks.append(
                module.register_forward_hook(
                    attention_hook(name)
                )
            )

    with torch.no_grad():
        features = model(input_tensor)

    for hook in hooks:
        hook.remove()

    for name, activation in activations.items():

        save_feature_visualizations(
            name,
            activation,
            output_dir
        )

    for name, attention in attention_weights.items():

        save_attention_visualization(
            name,
            attention,
            output_dir
        )

    print("Feature vector shape:", tuple(features.shape))
    print("Feature maps saved:", len(activations))
    print("Attention maps saved:", len(attention_weights))
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()
