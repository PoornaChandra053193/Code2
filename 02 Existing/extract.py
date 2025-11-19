import os
import re
import torch
import random
from PIL import Image
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet50 = models.resnet50(pretrained=True)
resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-1])
resnet50.to(device)
resnet50.eval()
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
bioclinical_bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
bioclinical_bert.to(device)
bioclinical_bert.eval()
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def extract_image_features(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = image_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = resnet50(image_tensor).squeeze()
    return features.cpu()
def encode_question(question):
    inputs = tokenizer(question, return_tensors='pt', padding=True, truncation=True, max_length=64).to(device)
    with torch.no_grad():
        outputs = bioclinical_bert(**inputs)
        attention_mask = inputs['attention_mask']
        last_hidden_state = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
        sum_mask = mask_expanded.sum(1)
        mean_pooled = sum_embeddings / sum_mask
    return mean_pooled.cpu()
def Extract():
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
                if "chest" in question.lower() or "chest" in answer.lower():
                    matched.append((image_path, question, answer))
        return matched
    qa_path = qa_file_map[split]
    image_path = image_folder_map[split]
    qa_pairs = load_qa_pairs(qa_path)
    matched_data = match_with_images(qa_pairs, image_path)
    if len(matched_data) == 0:
        print("No chest-related QA pairs found.")
        return
    if len(matched_data) < 6:
        sampled = matched_data
    else:
        sampled = random.sample(matched_data, 6)
    for i, (img_path, question, answer) in enumerate(sampled, 1):
        print(f"\nSample {i}")
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        img_features = extract_image_features(img_path)
        print("Image Feature Vector (ResNet-50, shape={}):".format(img_features.shape))
        print(img_features.numpy())
        question_embedding = encode_question(question)
        print("Question Encoding (BioClinicalBERT, shape={}):".format(question_embedding.shape))
        print(question_embedding.numpy())
