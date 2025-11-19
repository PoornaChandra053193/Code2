import os, re, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import vgg16
from transformers import ElectraTokenizer, ElectraModel
def Fused():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "./Input/MedVQ"
    split = "Train"
    qa_file_map = {
        "Train": os.path.join(data_dir, "Train", "All_QA_Pairs_train.txt"),
        "Val": os.path.join(data_dir, "Val", "All_QA_Pairs_val.txt"),
        "Test": os.path.join(data_dir, "Test", "VQAMed2019_Test_Questions_w_Ref_Answers.txt")
    }
    image_folder_map = {
        "Train": os.path.join(data_dir, "Train", "Train_images"),
        "Val": os.path.join(data_dir, "Val", "Val_images"),
        "Test": os.path.join(data_dir, "Test", "VQAMed2019_Test_Images")
    }
    def load_qa_pairs(file_path):
        qa_pairs = []
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split('|')
                if len(parts) >= 3:
                    image_id = parts[0].strip()
                    question = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', parts[1].strip())
                    answer = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', parts[2].strip())
                    qa_pairs.append((image_id, question, answer))
        return qa_pairs
    def match_with_images(qa_pairs, image_folder):
        matched = []
        available_images = {f.lower(): f for f in os.listdir(image_folder)}
        for image_id, question, answer in qa_pairs:
            image_filename = f"{image_id.lower()}.jpg"
            if image_filename in available_images:
                image_path = os.path.join(image_folder, available_images[image_filename])
                matched.append((image_path, question, answer))
        return matched
    vgg = vgg16(pretrained=True).features.to(device).eval()
    tokenizer = ElectraTokenizer.from_pretrained("google/electra-base-discriminator")
    electra = ElectraModel.from_pretrained("google/electra-base-discriminator").to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    class MLT_BiLSTM_Fusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.img_proj = nn.Linear(512, 512)
            self.text_proj = nn.Linear(768, 512)
            self.bilstm = nn.LSTM(512, 256, num_layers=1, bidirectional=True, batch_first=True)
            encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        def forward(self, img_feat, text_feat):
            img_feat = self.img_proj(img_feat)  
            text_feat = self.text_proj(text_feat)  
            seq = torch.stack([img_feat, text_feat], dim=1)  
            lstm_out, _ = self.bilstm(seq)  
            transformed = self.transformer(lstm_out)  
            fused = transformed.mean(dim=1)  
            return fused
    fuser = MLT_BiLSTM_Fusion().to(device)
    qa_path = qa_file_map[split]
    image_path = image_folder_map[split]
    qa_pairs = load_qa_pairs(qa_path)
    matched_data = match_with_images(qa_pairs, image_path)
    sampled = matched_data if len(matched_data) < 6 else random.sample(matched_data, 6)
    plt.figure(figsize=(14, 18))
    for i, (img_path, q, a) in enumerate(sampled, 1):
        image = Image.open(img_path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            img_feat = vgg(img_tensor).mean([2, 3])
        tokens = tokenizer(q, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_feat = electra(**tokens).last_hidden_state[:, 0, :]
        with torch.no_grad():
            fused = fuser(img_feat, text_feat)
        fused_np = fused.squeeze().cpu().numpy()
        ax_img = plt.subplot(6, 2, 2 * i - 1)
        ax_img.imshow(mpimg.imread(img_path))
        ax_img.axis('off')
        ax_img.set_title(f"Image {i}", fontsize=10, loc='left')
        plt.text(
            0.5, -0.25,
            f"Q: {q}\nA: {a}",
            fontsize=9,
            ha='center',
            va='top',
            transform=ax_img.transAxes
        )
        ax_feat = plt.subplot(6, 2, 2 * i)
        ax_feat.imshow(fused_np.reshape(1, -1), aspect='auto', cmap='viridis')
        ax_feat.axis('off')
        ax_feat.set_title(f"Fused Feature Vector (Sample {i})", fontsize=9)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.8)
    plt.show()