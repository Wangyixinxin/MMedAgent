import numpy as np
import torch
import clip
from clip.model import CLIP
from torchvision.transforms.transforms import Compose
from PIL import Image
from typing import List


class ModalityClip:
    def __init__(self, modality:List[str])->int:
        device="cuda" if torch.cuda.is_available() else "cpu"
        self.device=device
        model,preprocess=clip.load('ViT-B/32',device=self.device)
        self.model = model
        self.modality = modality
        self.text=clip.tokenize(modality).to(self.device)
        self.preprocess=preprocess

    def identify(self, filename:str)->int:

        image=self.preprocess(Image.open(filename)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits_per_image, logits_per_text=self.model(image,self.text)
            probs=logits_per_image.softmax(dim=-1).cpu().numpy()
            max_index = np.argmax(probs, axis=1)[0]
            # print(f"This image is a {self.modality[max_index]}")

        return max_index

if __name__=="__main__":
    modality=["panoramic dental x-ray","chest x-ray", "knee mri","Mammography","knee x-ray"]
    identifier=ModalityClip(modality)
    # upload medical images and input the filename
    # index=identifier.identify("dental/periodontals/Subject No.186.jpg")
    index=identifier.identify("chest.jpg")


