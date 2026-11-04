import torch


def calculate_complexity(model, height=256, width=128):
    parameters = sum(parameter.numel() for parameter in model.parameters())
    macs = None
    flops = None

    try:
        from thop import profile
        device = next(model.parameters()).device
        dummy = torch.randn(1, 3, height, width, device=device)
        was_training = model.training
        model.eval()
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        if was_training:
            model.train()
        flops = 2 * macs
    except Exception as error:
        print(f"MAC/FLOP profiling unavailable: {error}")

    return {
        "parameters": int(parameters),
        "MACs": None if macs is None else float(macs),
        "FLOPs": None if flops is None else float(flops)
    }
