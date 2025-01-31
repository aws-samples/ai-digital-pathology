import os 
import functools
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image

class HOPTIMUSZero(nn.Module):
    def __init__(self, checkpoint=None):
        super(HOPTIMUSZero, self).__init__()
        self.model = timm.create_model("hf_hub:bioptimus/H-optimus-0", pretrained=True)
        self.model.eval()

        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.707223, 0.578729, 0.703617), 
                    std=(0.211883, 0.230117, 0.177517)
                ),
            ])
        
    def to(self, device):
        self.model = self.model.to(device)
        return self
    
    @torch.inference_mode()
    def forward(self, image):
        if isinstance(image, Image.Image):
            image = self.transform(image).unsqueeze(dim=0)
            image = image.to("cuda" if torch.cuda.is_available() else "cpu")

        output = self.model.forward_features(image)
        return {
            'x_norm_cls_token': output[:, 0, :], 
            'x_norm_patch_tokens': output[:, 5:, :]
        } 

if __name__ == "__main__":
    # Initialize the model
    model = HOPTIMUSZero()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device)

    # Create a random input image
    input_image = torch.rand(3, 224, 224)
    input_image = transforms.ToPILImage()(input_image)
    
    # Perform inference
    output = model(input_image)
    print(output)
