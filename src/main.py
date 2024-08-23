import torch
from Training import train_ddpm
from Sample import sample
from Dataset_module import SRDataset
from Diffusion_Model import DiffusionModel
from PIL import Image
import torchvision.transforms as transforms
import os
import time

def main():
    start_time = time.time()  # Start timing the process
    print("Starting run...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = "/home/ofirn/home/ofirn/NewWetProject/All"

    # Train the model
    print("Starting training...")
    train_ddpm(time_steps=2000, epochs=30, batch_size=16, device=device,
               image_dims=(3, 128, 128), low_res_dims=(3, 32, 32))

    # Load the trained model
    model = DiffusionModel(time_steps=2000).to(device)
    model.load_state_dict(torch.load('/home/ofirn/home/ofirn/NewWetProject/checkpoints_AllOld_30epoch/ddpm_checkpoint_final.pth', map_location=device))
    model.eval()
    ###################################################
    # From here is Continue training the model on the second dataset
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    #dataset_path = "/home/ofirn/home/ofirn/NewWetProject/All"

    # Train the model
   # print("Starting Second training...")
   # train_ddpm(time_steps=2000, epochs=20, batch_size=16, device=device,
    #           image_dims=(3, 128, 128), low_res_dims=(3, 32, 32))


    ##################################################################
    # Load a high-resolution image (128x128)
    hr_image_path = "/home/ofirn/home/ofirn/NewWetProject/Mountain_24992.jpg"  # Replace with the actual path to your high-res image
    hr_img = Image.open(hr_image_path).convert("RGB")

    # Downscale the high-resolution image to 32x32
    transform_downscale = transforms.Compose([
        transforms.Resize((32, 32)),  # Downscale to 32x32
        transforms.ToTensor(),
    ])
    lr_img = transform_downscale(hr_img).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Save the downscaled low-resolution image
    transform_to_pil = transforms.ToPILImage()
    lr_img_pil = transform_to_pil(lr_img.squeeze(0).cpu())  # Remove batch dimension and convert to PIL
    low_res_image_path = "/home/ofirn/home/ofirn/NewWetProject/lr_Images"  # Replace with the directory where you want to save the low-res images
    os.makedirs(low_res_image_path, exist_ok=True)
    low_res_image_name = f"low_res_{int(time.time())}.jpeg"
    low_res_image_full_path = os.path.join(low_res_image_path, low_res_image_name)
    lr_img_pil.save(low_res_image_full_path)
    print(f"Low-resolution image saved to {low_res_image_full_path}")

    # Generate a unique filename for the output image
    output_dir = "/home/ofirn/home/ofirn/NewWetProject"  # Replace with the directory where you want to save the images
    os.makedirs(output_dir, exist_ok=True)
    output_image_name = f"sr_Mountain_train_AllOld_30epoch.jpeg"  # Use timestamp to generate unique filename
    output_image_path = os.path.join(output_dir, output_image_name)

    print("Starting image generation...")
    sample(model, lr_img, device=device, output_path=output_image_path)
    print(f"Image generation completed and saved to {output_image_path}")

    end_time = time.time()  # End timing the process
    total_time = end_time - start_time  # Calculate the total time taken
    print(f"Total process time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
