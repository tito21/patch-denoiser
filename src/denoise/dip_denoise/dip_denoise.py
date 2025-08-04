import torch
from tqdm.auto import tqdm

device = torch.device("mps")

def get_model(shape, out_channels=7):
    model = torch.nn.Sequential(
        torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
        torch.nn.InstanceNorm2d(32),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
        torch.nn.InstanceNorm2d(32),
        torch.nn.ReLU(),
        torch.nn.Upsample(scale_factor=2),

        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
        torch.nn.InstanceNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
        torch.nn.InstanceNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.Upsample(scale_factor=2),

        torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
        torch.nn.InstanceNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
        torch.nn.InstanceNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.Upsample(scale_factor=2),

        torch.nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
        torch.nn.Upsample(size=shape),
        torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),

        # torch.nn.Tanh()
    ).to(device)
    return model

def dip_denoise(data, iterations=1000, learning_rate=1e-4):

    data = torch.tensor(data, dtype=torch.float32, device=device)
    model = get_model(data.shape[:2], out_channels=data.shape[-1])
    data = data.permute(2, 0, 1).unsqueeze(0)  # Reshape to (1, C, H, W)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    criterion = torch.nn.MSELoss().to(device)

    z = torch.randn(1, 32, 16, 32).to(device)

    pbar = tqdm(total=iterations, desc="Training Progress")
    pbar.set_postfix({"loss": 0.0})
    for epoch in range(iterations):
        model.train()
        optimizer.zero_grad()
        output = model(z)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"loss": loss.item()})
        pbar.update(1)
    pbar.close()

    output = model(z)
    return output.squeeze().cpu().detach().numpy()