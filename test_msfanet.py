import torch
from torchreid.models import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model(
    name="msfanet",
    num_classes=751,
    loss="triplet",
    pretrained=False
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)

print("Device:", device)
print("Total parameters:", total_params)
print("Total parameters (M):", total_params / 1e6)
print("Trainable parameters:", trainable_params)

x = torch.randn(2, 3, 256, 128).to(device)
labels = torch.tensor([0, 1]).to(device)

model.train()
logits, features = model(x)

print("Training logits:", logits.shape)
print("Training features:", features.shape)

loss = torch.nn.functional.cross_entropy(logits, labels)
loss.backward()

print("Backward pass: successful")

model.eval()

with torch.no_grad():
    output = model(x)

print("Evaluation features:", output.shape)

maps = model.get_visual_maps()
print("Visual maps:", len(maps))

print("MSFANet test completed")
