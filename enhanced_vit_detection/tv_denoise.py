import torch
from skimage.restoration import denoise_tv_chambolle
from torchvision.transforms import functional as TF

class TVDenoise:
    @staticmethod
    def apply_denoise(images):
        denoised_images = []
        for image in images:
            image_np = image.detach().cpu().numpy()
            denoised_image_np = denoise_tv_chambolle(image_np, weight=0.1, multichannel=True)
            denoised_image = TF.to_tensor(denoised_image_np).to(images.device)
            denoised_images.append(denoised_image)
        return torch.stack(denoised_images)