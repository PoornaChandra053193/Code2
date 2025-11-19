import os
import re
import random
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision.transforms as transforms
from torchvision.models import vgg16
from transformers import ElectraTokenizer, ElectraModel
from skimage.feature import local_binary_pattern
def Extract():
    data_dir = "./Input/MedVQ"
    split = "Train"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    def extract_color_histogram(img, bins=(8, 8, 8)):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0,180, 0,256, 0,256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    def extract_lbp_features(gray_img):
        lbp = local_binary_pattern(gray_img, P=8, R=1, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        return hist
    def extract_edge_features(gray_img):
        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        return magnitude.flatten()[:100]
    tokenizer = ElectraTokenizer.from_pretrained("google/electra-base-discriminator")
    electra = ElectraModel.from_pretrained("google/electra-base-discriminator").to(device)
    vgg = vgg16(pretrained=True).features.to(device)
    vgg.eval()
    vgg_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    qa_path = qa_file_map[split]
    image_path = image_folder_map[split]
    qa_pairs = load_qa_pairs(qa_path)
    matched_data = match_with_images(qa_pairs, image_path)
    sampled = matched_data if len(matched_data) < 6 else random.sample(matched_data, 6)
    for i, (img_path, question, answer) in enumerate(sampled, 1):
        img_raw = mpimg.imread(img_path)
        img_cv2 = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_tensor = vgg_transform(img_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            vgg_features = vgg(img_tensor).squeeze().cpu().numpy()
        inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            text_features = electra(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        color_feat = extract_color_histogram(img_rgb)
        lbp_feat = extract_lbp_features(img_gray)
        edge_feat = extract_edge_features(img_gray)
        print(f"\n📸 {os.path.basename(img_path)}")
        print(f"🔹 VGG16 Feature Shape: {vgg_features.shape}")
        print(f"🔹 Color Histogram Shape: {color_feat.shape}")
        print(f"🔹 LBP (Texture) Shape: {lbp_feat.shape}")
        print(f"🔹 Edge Features Shape: {edge_feat.shape}")
        print(f"📝 ELECTRA (Text) Shape: {text_features.shape}")
        fig, axs = plt.subplots(1, 4, figsize=(18, 4))
        fig.suptitle(f"Features of: {os.path.basename(img_path)}", fontsize=12)
        axs[0].imshow(img_rgb)
        axs[0].axis('off')
        axs[0].set_title(f"Q: {question}\nA: {answer}", fontsize=8)
        axs[1].plot(color_feat[:100])
        axs[1].set_title("Color Histogram")
        axs[1].set_xlabel("Bin Index")
        axs[1].set_ylabel("Frequency")
        axs[2].bar(range(len(lbp_feat)), lbp_feat)
        axs[2].set_title("LBP Texture")
        axs[2].set_xlabel("Pattern Index")
        axs[2].set_ylabel("Frequency")
        combined_feat = np.concatenate((vgg_features.flatten()[:50], text_features.flatten()[:50]))
        axs[3].plot(combined_feat)
        axs[3].set_title("VGG + ELECTRA Summary")
        axs[3].set_xlabel("Feature Index")
        axs[3].set_ylabel("Activation")
        plt.tight_layout()
        plt.show()