from time import time
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from Diffusion_Model import DiffusionModel
from Dataset_module import SRDataset
from Sample import sample  # Assuming your sample function is in Sample.py
from PIL import Image
import torchvision.transforms as transforms
import os



def train_ddpm(time_steps=2000, epochs=20, batch_size=16, device=None, image_dims=(3, 128, 128),
               low_res_dims=(3, 32, 32)):
    # Determine the device to use (GPU if available, otherwise CPU)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the diffusion model with the specified number of timesteps.
    ddpm = DiffusionModel(time_steps=time_steps)

    # Extract dimensions from the high-resolution and low-resolution images.
    c, hr_sz, _ = image_dims
    _, lr_sz, _ = low_res_dims

    # Load the dataset containing high-res and low-res images.
    dataset_path = r"/home/ofirn/home/ofirn/NewWetProject/noise_Train"
    ds = SRDataset(dataset_path, hr_sz=hr_sz, lr_sz=lr_sz)
    print(f"Dataset size: {len(ds)} images")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    print(f"DataLoader initialized with batch size: {batch_size}")

    # Set up the optimizer for the diffusion model.
    opt = torch.optim.Adam(ddpm.model.parameters(), lr=1e-3)
    # Define the loss function.
    criterion = nn.MSELoss(reduction="mean")

    # Move the model to the appropriate device (GPU or CPU).
    ddpm.model.to(device)

    if not os.path.exists('./checkpoints_noise_16Batch'):
        os.makedirs('./checkpoints_noise_16Batch')

    for ep in range(epochs):
        ddpm.model.train()
        print(f"Epoch {ep}:")
        losses = []
        stime = time()

        for i, (x, y) in enumerate(loader):
           # print(f"Processing batch {i}, batch size: {x.shape[0]}")
            bs = y.shape[0]
            x, y = x.to(device), y.to(device)

           # print(f"x shape: {x.shape}, y shape: {y.shape}")

            ts = torch.randint(low=1, high=ddpm.time_steps, size=(bs,))
            gamma = ddpm.alpha_hats[ts].to(device)
            ts = ts.to(device=device)

            y, target_noise = ddpm.add_noise(y, ts)
            x_upsampled = nn.functional.interpolate(x, size=(hr_sz, hr_sz), mode='bicubic', align_corners=False)
          #  print(f"x_upsampled shape: {x_upsampled.shape}, y shape: {y.shape}")

            y = torch.cat([x_upsampled, y], dim=1)
            predicted_noise = ddpm.model(y, gamma)
            loss = criterion(target_noise, predicted_noise)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Loss is NaN or inf at batch {i}, epoch {ep}")
                print(f"target_noise: {target_noise}")
                print(f"predicted_noise: {predicted_noise}")
                continue

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

            if i % 250 == 0:
                print(f"Loss: {loss.item()}; step {i}; epoch {ep}")

            # Save a generated image every 500 steps
            if i % 500 == 0:
                test_input_image = Image.open("/home/ofirn/home/ofirn/NewWetProject/low_res.jpeg")
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),  # Resize to the expected input size
                    transforms.ToTensor(),  # Convert the image to a tensor
                ])
                test_input_image = transform(test_input_image).unsqueeze(0).to(device)
                output_image_path = f"./checkpoints_noise_16Batch/epoch_{ep}_step_{i}_sample.jpeg"
                sample(ddpm, test_input_image, device=device, output_path=output_image_path)

        ftime = time()
        avg_loss = sum(losses) / len(losses) if losses else float('inf')
        print(f"Epoch trained in {ftime - stime}s; Avg loss => {avg_loss}")

        # Save the model checkpoint after each epoch
        torch.save(ddpm.state_dict(), f"./checkpoints_noise_16Batch/sr_ep_{ep}.pt")

        # Save a generated image after each epoch
        test_input_image = Image.open("/home/ofirn/home/ofirn/NewWetProject/low_res.jpeg")
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # Resize to the expected input size
            transforms.ToTensor(),  # Convert the image to a tensor
        ])
        test_input_image = transform(test_input_image).unsqueeze(0).to(device)
        output_image_path = f"./checkpoints_noise_16Batch/epoch_{ep}_sample.jpeg"
        sample(ddpm, test_input_image, device=device, output_path=output_image_path)
    torch.save(ddpm.state_dict(), './checkpoints_noise_16Batch/ddpm_checkpoint_final.pth')


if __name__ == "__main__":
    train_ddpm()
