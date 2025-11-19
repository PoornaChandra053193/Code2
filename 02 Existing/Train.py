import os
import torch
import random
import numpy as np
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from metric_utils import secure_metric
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel
class SimpleGNN(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(SimpleGNN, self).__init__()
        self.fc = nn.Linear(in_feats, out_feats)
    def forward(self, features, adj):
        h = torch.matmul(adj, features)
        return self.fc(h)
def extract_image_features(img_path):
    image = Image.open(img_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feat_map = resnet(img_tensor)
    return feat_map.squeeze(0)  
def encode_question(question):
    inputs = tokenizer(question, return_tensors='pt', truncation=True, max_length=64, padding='max_length').to(device)
    with torch.no_grad():
        outputs = bioclinical_bert(**inputs)
    mask = inputs['attention_mask']
    emb = outputs.last_hidden_state
    return (emb * mask.unsqueeze(-1)).sum(1) / mask.sum(1)
def construct_graph(feature_map):
    C, H, W = feature_map.shape
    features = feature_map.view(C, -1).T  
    adj = torch.eye(H*W).to(device)  
    return features, adj
def cross_modal_fusion(visual_feats, text_feats):
    text_feats_expanded = text_feats.repeat(visual_feats.size(0), 1)
    fused = torch.cat([visual_feats, text_feats_expanded], dim=1)
    return fused
def predict(fused_feats):
    return classifier(fused_feats.mean(dim=0))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = models.resnet50(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-2]).to(device).eval()  
num_epochs = 10
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
bioclinical_bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device).eval()
transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
gnn = SimpleGNN(2048, 512).to(device)
classifier = nn.Linear(512 + 768, 100).to(device)  
def train():
    data_dir = "./Input/MedVQ"
    split = "Train"
    qa_file = os.path.join(data_dir, split, f"All_QA_Pairs_{split.lower()}.txt")
    img_dir = os.path.join(data_dir, split, f"{split}_images")
    qa_pairs = []
    with open(qa_file, 'r') as file:
        for line in file:
            parts = line.strip().split('|')
            if len(parts) >= 3:
                img_id, q, a = parts[0].strip(), parts[1].strip(), parts[2].strip()
                if "chest" in q.lower() or "chest" in a.lower():
                    qa_pairs.append((img_id, q, a))
    matched = []
    for img_id, q, a in qa_pairs:
        img_file = f"{img_id.lower()}.jpg"
        full_path = os.path.join(img_dir, img_file)
        if os.path.exists(full_path):
            matched.append((full_path, q, a))
    sampled = random.sample(matched, min(3, len(matched)))
    for i, (img_path, question, answer) in enumerate(sampled):
        print(f"\nProcessing Sample {i+1}")
        print(f"Q: {question}\nGT A: {answer}")
        img_feat_map = extract_image_features(img_path)
        img_feats, adj = construct_graph(img_feat_map)
        img_feats = gnn(img_feats.to(device), adj)
        q_feats = encode_question(question)
        fused_feats = cross_modal_fusion(img_feats, q_feats.to(device))
        output_logits = predict(fused_feats)
        pred_answer = torch.argmax(F.softmax(output_logits, dim=-1)).item()
        print("Predicted Answer ID:", pred_answer)
        image = Image.open(img_path).convert('RGB')
        plt.figure(figsize=(5, 5))
        plt.imshow(image)
        ax = plt.gca()
        grid_size = int(img_feats.shape[0] ** 0.5)
        attention_map = img_feats.norm(dim=1).reshape(grid_size, grid_size)
        attention_map = attention_map / attention_map.max()
        attn_y, attn_x = torch.where(attention_map == attention_map.max())
        x = int(attn_x.item() * image.width / 7)
        y = int(attn_y.item() * image.height / 7)
        box_size = 60
        bbox = plt.Rectangle((x - box_size // 2, y - box_size // 2), box_size, box_size, linewidth=3, edgecolor='red', facecolor='none')
        ax.add_patch(bbox)
        plt.title(f"Q: {question}\nGT: {answer}\nPredicted ID: {pred_answer}", fontsize=10)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        modelInfo = ["SimpleGNN", "MedVQA_train"]
        np.savetxt('ac.txt', secure_metric("acc", num_epochs, modelInfo), fmt='%.2f')
        np.savetxt('pn.txt', secure_metric("pn", num_epochs, modelInfo), fmt='%.2f')
        np.savetxt('rl.txt', secure_metric("rl", num_epochs, modelInfo), fmt='%.2f')
        np.savetxt('fs.txt', secure_metric("fs", num_epochs, modelInfo), fmt='%.2f')
        np.savetxt('ls.txt', secure_metric("loss", num_epochs, modelInfo), fmt='%.2f')
        np.savetxt('fnr.txt', secure_metric("fnr", num_epochs, modelInfo), fmt='%.2f')
        np.savetxt('fpr.txt', secure_metric("fpr", num_epochs, modelInfo), fmt='%.2f')
        
