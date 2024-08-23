import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim
from sklearn.metrics import mean_squared_error as compare_mse
from Dataset_module_old import SRDataset
from Diffusion_Model_old import DiffusionModel
from Sample import sample
import torchvision.transforms as transforms


class CustomTestDataset(Dataset):
    def __init__(self, test_data_path):
        self.test_data_path = test_data_path
        self.images = sorted([f for f in os.listdir(test_data_path) if f.endswith(('.jpg', '.jpeg', '.png'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.test_data_path, img_name)
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)

        return img_name, img

def test_model(model_path, test_data_path, device=None, batch_size=1, image_dims=(3, 128, 128)):
    # Determine the device to use (GPU if available, otherwise CPU)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the pre-trained model
    ddpm = DiffusionModel(time_steps=2000)  # Ensure this matches your training configuration
    ddpm.load_state_dict(torch.load(model_path, map_location=device))
    ddpm.model.to(device)
    ddpm.model.eval()

    # Prepare the test dataset
    dataset = CustomTestDataset(test_data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    print(f"Testing on {len(dataset)} images")

    # Directory to save test results
    results_dir = r'/home/ofirn/home/ofirn/NewWetProject/checkpoints_AllOLD_16Batch_newAugm/Test'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Initialize lists to store PSNR, SSIM, and MSE values
    psnr_values = []
    ssim_values = []
    mse_values = []

    # Test loop
    with torch.no_grad():
        for i, (img_name, img) in enumerate(loader):
            img_name = img_name[0]  # Extract the image name from the batch
            if "32" in img_name:
                # Process low-resolution images
                lr_img = Image.fromarray(img.squeeze(0).numpy())
                lr_img = transforms.ToTensor()(lr_img).unsqueeze(0).to(device)

                # Generate and save the high-resolution image
                output_image_name = img_name.replace('32', 'generated')  # Modify name for generated image
                output_image_path = os.path.join(results_dir, output_image_name)
                sample(ddpm, lr_img, device=device, output_path=output_image_path)
                print(f"Saved test result to {output_image_path}")

                # Load the generated image and the corresponding high-resolution original
                generated_hr_img = Image.open(output_image_path).convert('RGB')
                generated_hr_img = np.array(generated_hr_img)
                corresponding_hr_img_name = img_name.replace('32', '128')
                original_hr_img_path = os.path.join(test_data_path, corresponding_hr_img_name)
                original_hr_img = Image.open(original_hr_img_path).convert('RGB')
                original_hr_img = np.array(original_hr_img)

                # Compute PSNR, SSIM, and MSE
                psnr_value = compare_psnr(original_hr_img, generated_hr_img, data_range=255)
                ssim_value = compare_ssim(original_hr_img, generated_hr_img, data_range=255, multichannel=True, win_size=3)
                mse_value = compare_mse(original_hr_img.flatten(), generated_hr_img.flatten())

                # Append the values for each image
                psnr_values.append(psnr_value)
                ssim_values.append(ssim_value)
                mse_values.append(mse_value)

                print(f"Image {i + 1} - PSNR: {psnr_value:.4f}, SSIM: {ssim_value:.4f}, MSE: {mse_value:.4f}")

    # Compute and print the average PSNR, SSIM, and MSE
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_mse = np.mean(mse_values)
    print(f"Average PSNR: {avg_psnr:.4f}, Average SSIM: {avg_ssim:.4f}, Average MSE: {avg_mse:.4f}")

    print("Testing completed.")

if __name__ == "__main__":
    # Set the path to your saved model and test dataset
    model_path = r'/home/ofirn/home/ofirn/NewWetProject/checkpoints_AllOLD_16Batch_newAugm/ddpm_checkpoint_final.pth'
    test_data_path = r"/home/ofirn/home/ofirn/NewWetProject/test-original-resolution"

    # Run the test script
    test_model(model_path, test_data_path)
