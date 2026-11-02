
import torch


def calculate_complexity(model, height=256, width=128):
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    result = {
        "Total Parameters (M)": total_parameters / 1e6,
        "Trainable Parameters (M)": trainable_parameters / 1e6,
        "MACs (G)": None,
        "FLOPs (G)": None
    }

    try:
        from thop import profile

        device = next(model.parameters()).device
        dummy_input = torch.randn(
            1, 3, height, width,
            device=device
        )

        was_training = model.training
        model.eval()

        macs, _ = profile(
            model,
            inputs=(dummy_input,),
            verbose=False
        )

        if was_training:
            model.train()

        result["MACs (G)"] = macs / 1e9
        result["FLOPs (G)"] = (2 * macs) / 1e9

    except Exception as error:
        print("MAC/FLOP calculation unavailable:", error)

    return result
