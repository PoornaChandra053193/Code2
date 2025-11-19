import os
import re
import torch
import random
from PIL import Image
import torch.nn.functional as F
from torchvision.models import vgg16
import torchvision.transforms as transforms
from transformers import ElectraTokenizer, ElectraModel
def Dist_Attention():
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
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    vgg = vgg16(pretrained=True).features.to(device).eval()
    tokenizer = ElectraTokenizer.from_pretrained("google/electra-base-discriminator")
    electra = ElectraModel.from_pretrained("google/electra-base-discriminator").to(device).eval()
    text_proj = torch.nn.Linear(768, 512).to(device)
    ans_proj = torch.nn.Linear(768, 512).to(device)
    def attention_fusion(image_feat, text_feat):
        text_feat_proj = text_proj(text_feat)
        image_feat = F.normalize(image_feat, dim=1)
        text_feat_proj = F.normalize(text_feat_proj, dim=1)
        scores = torch.matmul(image_feat, text_feat_proj.T)
        attention_weights = F.softmax(scores, dim=1)
        fused = torch.matmul(attention_weights, text_feat_proj)
        return fused, attention_weights
    def distill_answer_embedding(fused_feat, answer_emb):
        distill = torch.mean(torch.stack([fused_feat, answer_emb]), dim=0)
        return distill
    qa_path = qa_file_map[split]
    image_path = image_folder_map[split]
    qa_pairs = load_qa_pairs(qa_path)
    matched_data = match_with_images(qa_pairs, image_path)
    sampled = matched_data if len(matched_data) < 6 else random.sample(matched_data, 6)
    feature_bank = []
    print(f"\n🚀 Starting attention-based feature fusion on {len(sampled)} samples...\n")
    for i, (img_path, q, a) in enumerate(sampled, 1):
        print(f"\n=== Sample {i} ===")
        print(f"🔹 Image Path: {img_path}")
        print(f"❓ Question: {q}")
        print(f"✅ Answer: {a}")
        image = Image.open(img_path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            img_feat = vgg(img_tensor).mean([2, 3])
        print(f"📸 Image Feature Shape: {img_feat.shape}")
        tokens = tokenizer(q, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_feat = electra(**tokens).last_hidden_state[:, 0, :]
        print(f"📝 Text Feature Shape: {text_feat.shape}")
        ans_tokens = tokenizer(a, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            ans_feat = electra(**ans_tokens).last_hidden_state[:, 0, :]
            ans_feat_proj = ans_proj(ans_feat)
        print(f"🗣️ Answer Embedding Shape: {ans_feat_proj.shape}")
        with torch.no_grad():
            fused_feat, attn_weights = attention_fusion(img_feat, text_feat)
            final_rep = distill_answer_embedding(fused_feat, ans_feat_proj)
        print(f"💡 Attention Weights: {attn_weights.cpu().numpy()}")
        print(f"🔗 Fused Feature (first 5 dims): {fused_feat.squeeze().cpu().numpy()[:5]}")
        print(f"🧠 Distilled Feature (first 5 dims): {final_rep.squeeze().cpu().numpy()[:5]}")
        feature_bank.append({
            'img': img_feat.squeeze().cpu().numpy(),
            'text': text_feat.squeeze().cpu().numpy(),
            'answer': ans_feat_proj.squeeze().cpu().numpy(),
            'fused': final_rep.squeeze().cpu().numpy()
        })