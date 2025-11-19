import re
import os
import random
import shutil
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
def Collect():
    root = tk.Tk()
    root.withdraw()  
    selected_dir = filedialog.askdirectory(title="Select the dataset directory")
    if not selected_dir:
        print("No directory selected. Exiting.")
        return
    dir_name = os.path.basename(selected_dir)
    dest_dir = os.path.join("Input", dir_name)
    if not os.path.exists(dest_dir):
        shutil.copytree(selected_dir, dest_dir)
        print(f"Copied {selected_dir} to {dest_dir}")
    else:
        print(f"Directory {dest_dir} already exists. Using existing data.")
    data_dir = dest_dir
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
    qa_path = qa_file_map[split]
    image_path = image_folder_map[split]
    qa_pairs = load_qa_pairs(qa_path)
    matched_data = match_with_images(qa_pairs, image_path)
    if len(matched_data) < 6:
        sampled = matched_data
    else:
        sampled = random.sample(matched_data, 6)
    plt.figure(figsize=(15, 10))
    for i, (img_path, q, a) in enumerate(sampled, 1):
        img = mpimg.imread(img_path)
        plt.subplot(2, 3, i)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(f"Q: {q}\nA: {a}", fontsize=10)
    plt.tight_layout()
    plt.show()
