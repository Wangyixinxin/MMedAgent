import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import json
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
from tqdm import tqdm 
model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
data_path = './data_for_classification_2'
# Dataset class
class BiomedImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        for subfolder in os.listdir(root_dir):
            subdir = os.path.join(root_dir, subfolder)
            if os.path.isdir(subdir):
                for filename in os.listdir(subdir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(subdir, filename))
                        self.labels.append(subfolder)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, label, image_path
    
transform = transforms.Compose([
    preprocess
])


dataset = BiomedImageDataset(root_dir=data_path, transform=transform)
data_loader = DataLoader(dataset, batch_size=10, shuffle=False)

# build label list

labels = [
    'adenocarcinoma histopathology',
    'brain MRI',
    'covid line chart',
    'diagnostic flowchart',
    'diagnostic scatter plot',
    'squamous cell carcinoma histopathology',
    'immunohistochemistry histopathology',
    'bone X-ray',
    'chest X-ray',
    'abdomen CT',
    'lung CT',
    'pie chart',
    'hematoxylin and eosin histopathology',
    'gross'
]

def predict_and_save(data_loader, model, tokenizer, device, labels):
    results = []
    template = 'the photo can be classified as '
    with torch.no_grad():
        for images, _, paths in tqdm(data_loader, desc="Predicting"):
            images = images.to(device)
            texts = tokenizer([template + l for l in labels], context_length=256).to(device)
            image_features, text_features, logit_scale = model(images, texts)
            logits = (logit_scale * image_features @ text_features.t()).softmax(dim=-1)
            top_preds = torch.topk(logits, 3).indices
            
            for path, top_pred in zip(paths, top_preds):
                result = {
                    'image_path': path,
                    'predicted_labels': [labels[idx] for idx in top_pred.tolist()]
                }
                results.append(result)
    
    with open('prediction1.json', 'w') as f:
        json.dump(results, f, indent=4)

predict_and_save(data_loader, model, tokenizer, device, labels)