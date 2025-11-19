import os
import re
import cv2 
import torch
import random
import hashlib
import numpy as np
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoModelForMaskedLM
def Classify():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "./Dataset/MedVQ"
    split = "Train"
    epochs = 10
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
    resnet = models.resnet152(pretrained=True).to(device).eval()
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_mlm = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    qa_path = qa_file_map[split]
    image_path = image_folder_map[split]
    qa_pairs = load_qa_pairs(qa_path)
    matched_data = match_with_images(qa_pairs, image_path)
    sampled = matched_data if len(matched_data) < 6 else random.sample(matched_data, 6)
    plt.figure(figsize=(15, 10))
    display_indices = set(random.sample(range(len(sampled)), 6))
    reference_feature = None
    abnormal_keywords = [
        "abnormal", "abnormality", "finding", "alarming", "pathology", "mass", "tumor",
        "disease", "lesion", "condition", "nodule", "opacity", "fracture", "infiltrate",
        "effusion", "collapse", "consolidation", "infection", "pneumonia", "cancer",
        "malignancy", "hemorrhage", "edema", "degeneration", "tear", "rupture", "necrosis",
        "enlargement", "swelling", "dislocation", "foreign body", "calcification",
        "clot", "obstruction", "inflammation", "scar", "fibrosis", "stenosis", "abscess",
        "metastasis", "erosion", "dilatation", "atrophy", "ischemia", "cyst", "sclerosis",
        "anomaly", "thrombus", "infection", "congestion", "growth", "malformation",
        "hyperplasia", "aneurysm", "embolism", "fistula", "granuloma", "hyperintensity",
        "hypodensity", "shadow", "plaque", "air-fluid level", "distension", "collapse"
    ]
    for i, (img_path, q, a) in enumerate(sampled, 1):
        image = Image.open(img_path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = resnet(img_tensor).squeeze()
        if reference_feature is None:
            reference_feature = features
        similarity = F.cosine_similarity(reference_feature.unsqueeze(0), features.unsqueeze(0)).item()
        predicted_text = ""
        confidence = None
        if any(word in q.lower() for word in abnormal_keywords):
            q_clean = q[:-1] if q.endswith("?") else q
            pattern = r"(" + "|".join(re.escape(word) for word in abnormal_keywords) + r")"
            masked_prompt = re.sub(pattern, "the [MASK]", q_clean, flags=re.IGNORECASE)
            if "[MASK]" not in masked_prompt:
                masked_prompt += " the [MASK]"
            inputs = tokenizer(masked_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = bert_mlm(**inputs)
            predictions = outputs.logits
            masked_indices = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            predicted_tokens = []
            for idx in masked_indices:
                probs = F.softmax(predictions[0, idx], dim=-1)
                top_prob, top_token_id = probs.topk(1)
                token_str = tokenizer.convert_ids_to_tokens(top_token_id.squeeze().item())
                predicted_tokens.append(token_str)
                confidence = round(top_prob.squeeze().item(), 4)
            predicted_text = tokenizer.convert_tokens_to_string(predicted_tokens)
        print(f"\nSample {i}:")
        print("Image Path:", img_path)
        print("Question:", q)
        print("Answer:", a)
        print("Predicted Text:", predicted_text)
        if confidence is not None:
            print("Confidence Score:", confidence)
        print("Similarity with Image 1:", round(similarity, 3))
        img = mpimg.imread(img_path)
        if img.ndim == 2:
            img = np.stack((img,) * 3, axis=-1)
        if any(word in q.lower() for word in abnormal_keywords):
            img_h, img_w = img.shape[:2]
            box_width, box_height = int(img_w * 0.5), int(img_h * 0.5)
            start_x = int((img_w - box_width) / 2)
            start_y = int((img_h - box_height) / 2)
            end_x = start_x + box_width
            end_y = start_y + box_height
            img = img.copy()    
            if img.dtype == np.uint8:
                if confidence is not None:
                    if confidence > 0.7:
                        box_color = (0, 255, 0)       
                        fill_color = (144, 238, 144)  
                    elif confidence > 0.4:
                        box_color = (255, 255, 0)     
                        fill_color = (255, 255, 153)  
                    else:
                        box_color = (255, 0, 0)       
                        fill_color = (255, 182, 193)  
                else:
                    box_color = (255, 0, 0)
                    fill_color = (255, 182, 193)
            else:
                box_color = (1, 0, 0)
                fill_color = (1, 0.7, 0.7)
            overlay = img.copy()
            cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), fill_color, -1)
            alpha = 0.3
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
            thickness = 6
            cv2.rectangle(img, (start_x, start_y), (end_x, end_y), box_color, thickness)
        plt.subplot(2, 3, i)
        plt.imshow(img)
        plt.axis('off')
        text_color = 'white'
        if confidence is not None:
            if confidence > 0.7:
                text_color = 'lightgreen'
            elif confidence > 0.4:
                text_color = 'yellow'
            else:
                text_color = 'red'
        display_text = f"Q: {q}\nA: {a}"
        if predicted_text:
            display_text += f"\nPred: {predicted_text.strip()}\nConf: {confidence}"
        display_text += f"\nSim: {round(similarity, 3)}"
        plt.text(0, 10, display_text, fontsize=8, color=text_color, backgroundcolor='black', wrap=True)
    plt.tight_layout()
    plt.show()