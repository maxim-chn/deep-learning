import torch
import torchvision.utils as vutils


def sample(model, lr_img, device="cuda", output_path="sr_sample.jpeg"):
    model.to(device)
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        # Upsample the low-resolution image to match the high-resolution dimensions
        lr_img_upsampled = torch.nn.functional.interpolate(lr_img, size=(128, 128), mode='bicubic', align_corners=False)

        # Initialize random noise with the same size as the high-resolution image
        y = torch.randn(1, 3, 128, 128, device=device)

        # Move the upsampled low-res image to the device
        lr_img_upsampled = lr_img_upsampled.to(device)

        # Reverse the diffusion process
        for t in reversed(range(model.time_steps)):
            alpha_t, alpha_t_hat, beta_t = model.alphas[t].to(device), model.alpha_hats[t].to(device), model.betas[
                t].to(device)
            t = torch.tensor([t], device=device)  # Move t to device
            pred_noise = model(torch.cat([lr_img_upsampled, y], dim=1), t)  # Predict noise
            y = (torch.sqrt(1 / alpha_t)) * (y - (1 - alpha_t) / torch.sqrt(1 - alpha_t_hat) * pred_noise)
            if t > 1:
                y += torch.sqrt(beta_t) * torch.randn_like(y)

        # Normalize the final image for display
        y = (y - y.min()) / (y.max() - y.min())
        vutils.save_image(y, output_path)  # Save the generated image
