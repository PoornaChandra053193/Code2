"""

 =========================================================================================================================================================
==                                                                                                                                                       ==
==                              PRIVACY-PRESERVING FEDERATED LEARNING FOR MEDICAL VISUAL QUESTION ANSWERING IN HEALTHCARE SYSTEMS                        ==
==                                                                                                                                                       ==
==                                                                  PROPOSED APPROACH                                                                    ==
==                                                                                                                                                       ==
 =========================================================================================================================================================

"""

import os
import time
import logging
import warnings
import matplotlib
import customtkinter
from PIL import Image
from Train import train
from fused import Fused
from collect import Collect
from extract import Extract
from classify import Classify
from plyer import notification
from attention import Dist_Attention
from Metrics import PerformanceMetrics
from tkinter import Label , messagebox
matplotlib.use("TkAgg")
window = customtkinter.CTk()
warnings.filterwarnings("ignore")
os.makedirs("Output", exist_ok = True)
os.makedirs("Graphs", exist_ok = True)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger().setLevel(logging.ERROR)
RED = '\033[41m'
BOLD = '\033[1m'
RBOLD = '\033[22m'
GREEN = '\033[32m'
DEFAULT = '\033[39m'
MAGENTA =  '\033[35m'
Input_dir = "Input"
Output_dir = "Output"

os.makedirs(Input_dir, exist_ok=True)
os.makedirs(Output_dir, exist_ok=True)

def display_toast(title, message):
        max_title_length = 64  
        truncated_title = title[:max_title_length]
        notification.notify(title=truncated_title, message=message, timeout=1, app_icon="Assets/Icon.ico") # type: ignore

def Data_Collection():
    print (f"\n\t\t\t==========================************* {GREEN}DATA COLLECTION PROCESS{DEFAULT} ******************==============================\n")    
    display_toast("PRIVACY-PRESERVING FEDERATED LEARNING FOR MED-VQA", 'Data Collection is in Process....')
    time.sleep(2)
    print("\n Data Collection is Under Process......")
    Collect()
    print("\nData Collection Process Completed......\n")
    time.sleep(2)
    display_toast("PRIVACY-PRESERVING FEDERATED LEARNING FOR MED-VQA", 'Data Collection Process Completed....')
    time.sleep(3)
    display_toast("PRIVACY-PRESERVING FEDERATED LEARNING FOR MED-VQA", 'Next Click FEATURE EXTRACTION Button')

def Feature_Extraction():
    print (f"\n\t\t\t==========================************* {GREEN}FEATURE EXTRACTION PROCESS{DEFAULT} ******************==============================\n")    
    display_toast("PRIVACY-PRESERVING FEDERATED LEARNING FOR MED-VQA",'Feature Extraction is in Process....')
    time.sleep(2)
    print("\n Feature Extraction is Under Process......\n")
    time.sleep(2)
    Extract()
    time.sleep(2)
    display_toast("PRIVACY-PRESERVING FEDERATED LEARNING FOR MED-VQA",'Feature Extraction Process Completed....')
    time.sleep(3)
    print("\n Feature Extraction Process Completed......\n")
    display_toast("PRIVACY-PRESERVING FEDERATED LEARNING FOR MED-VQA",'Next DISTILLATION ATTENTION Button')

def Distillation_Attention():
    print (f"\n\t\t\t==========================************* {GREEN}DISTILLATION ATTENTION PROCESS {DEFAULT} *************==========================\n")
    display_toast("PRIVACY-PRESERVING FEDERATED LEARNING FOR MED-VQA",' Distillation Attention is in Process....')
    time.sleep(2)
    print("\n Distillation Attention is Under Process......\n")
    Dist_Attention()
    time.sleep(2)
    display_toast("PRIVACY-PRESERVING FEDERATED LEARNING FOR MED-VQA",' Distillation Attention Process Completed....')
    time.sleep(3)
    print("\n Distillation Attention Process Completed......\n")
    display_toast("PRIVACY-PRESERVING FEDERATED LEARNING FOR MED-VQA",'Next Click IMAGE & TEXT FUSION Button')

def Img_Txt_Fusion():
    print (f"\n\t\t\t==========================************* {GREEN}IMAGE & TEXT FUSION PROCESS{DEFAULT} *************==========================\n")
    display_toast("PRIVACY-PRESERVING FEDERATED LEARNING FOR MED-VQA",' Image & Text Fusion is in Process....')
    time.sleep(2)
    print("\n Image & Text Fusion is Under Process......\n")
    Fused()
    time.sleep(2)
    display_toast("PRIVACY-PRESERVING FEDERATED LEARNING FOR MED-VQA",' Image & Text Fusion Process Completed....')
    time.sleep(3)
    print("\n Image & Text Fusion Process Completed......\n")
    display_toast("PRIVACY-PRESERVING FEDERATED LEARNING FOR MED-VQA",'Next Click CLASSIFICATION Button')

def Classification():
    print (f"\n\t\t\t==========================************* {GREEN}CLASSIFICATION PROCESS{DEFAULT} *************==========================\n")
    display_toast("PRIVACY-PRESERVING FEDERATED LEARNING FOR MED-VQA",' Classification is in Process....')
    time.sleep(2)
    print("\n Classification is Under Process......\n")
    Classify()
    time.sleep(2)
    display_toast("PRIVACY-PRESERVING FEDERATED LEARNING FOR MED-VQA",' Classification Process Completed....')
    time.sleep(3)
    print("\n Classification Process Completed......\n")
    display_toast("PRIVACY-PRESERVING FEDERATED LEARNING FOR MED-VQA",'Next TRAINING Button')

def Training():
    print (f"\n\t\t\t==========================************* {GREEN}TRAINING PROCESS {DEFAULT} *************==========================\n")
    display_toast("PRIVACY-PRESERVING FEDERATED LEARNING FOR MED-VQA",' Training is in Process....')
    time.sleep(2)
    print("\n Training is Under Process......\n")
    train()
    time.sleep(2)
    display_toast("PRIVACY-PRESERVING FEDERATED LEARNING FOR MED-VQA",' Training Process Completed....')
    time.sleep(3)
    print("\n Training Process Completed......\n")
    display_toast("PRIVACY-PRESERVING FEDERATED LEARNING FOR MED-VQA",'Next PERFORMANCE METRICS Button')

def Performance_Metrics():
    print (f"\n\t==========================******************* {GREEN}PERFORMANCE METRICS{DEFAULT} *******************==========================\n")
    display_toast("PRIVACY-PRESERVING FEDERATED LEARNING FOR MED-VQA",' Performance Metrics is in Process....')
    time.sleep(2)
    print("\nPerformance Metrics is Under Process......\n")
    print("\nPlease check the Graphs in the Graphs Folder.....\n")
    messagebox.showinfo("Information","Performance metrics \nPlease check the Graphs in the Output/Graphs Folder.....")
    PerformanceMetrics()
    time.sleep(2)
    display_toast("PRIVACY-PRESERVING FEDERATED LEARNING FOR MED-VQA",' Performance Metrics Completed....')
    time.sleep(2)
    display_toast("PRIVACY-PRESERVING FEDERATED LEARNING FOR MED-VQA",'THANK YOU....')
    time.sleep(2)
    print("Performance Metrics Completed......\n")

def main():
    print(f"\n{BOLD}===================================================================************************************=======================================================================")
    print(f"\n\t\t\t\t\t{MAGENTA}PRIVACY-PRESERVING FEDERATED LEARNING FOR MEDICAL VISUAL QUESTION ANSWERING IN HEALTHCARE SYSTEMS{DEFAULT} \n")
    print(f"\n\t\t\t\t\t\t\t\t\t\t{GREEN}PROPOSED APPROACH{DEFAULT} \n")
    print(f"===================================================================************************************======================================================================={RBOLD}\n")
    image = Image.open("Assets/bg.png")
    background_image = customtkinter.CTkImage(image, size=(500, 50))
    def bg_resizer(e):
        if e.widget is window:
            i = customtkinter.CTkImage(image, size=(e.width, e.height))
            bg_lbl.configure(text="", image=i)
    bg_lbl = customtkinter.CTkLabel(window, text="", image=background_image)
    bg_lbl.place(x=0, y=0)
    window.title("PRIVACY-PRESERVING FEDERATED LEARNING FOR MED-VQA")
    window_width = 950
    window_height = 890
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    position_top = int(screen_height/2 - window_height/2)-35
    position_right = int(screen_width/2 - window_width/2)-100
    window.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')
    window.resizable(False, False)
    window.configure(fg_color="#ffffff")
    button_params = {
        'width': 320, 
        'height': 55, 
        'fg_color': '#ae6000', 
        'text_color': '#61ff69',
        'corner_radius': 50, 
        'hover_color': '#000000', 
        'border_color': '#ffee00',
        'border_width': 3, 
        'font': ('Georgia', 15, 'bold')
    }
    label_params = {
        'bg': window["background"], 
        'fg': '#000000', 
        'font': ('Georgia', 19, 'bold')
    }
    pad_label = Label(window, text="",bg="#7EF2FF")
    pad_label.pack()
    top_label = Label(window, text="PRIVACY-PRESERVING FEDERATED LEARNING FOR MEDICAL\n\nVISUAL QUESTION ANSWERING IN HEALTHCARE SYSTEMS",**label_params)
    top_label.pack(pady=18)
    button_frame = customtkinter.CTkFrame(window, width=390, height=620, corner_radius=90, fg_color="#a5da68", bg_color="#E9E9E9", background_corner_colors=("#ffffff", "#ffffff", "#EBEDD6", "#A2BC79"))
    button_frame.pack(pady=10, padx=20, expand=True)
    button_frame.pack_propagate(False)
    pad_label1 = Label(button_frame, text="",bg="#a5da68")
    pad_label1.pack()
    buttons = [
        ("DATA COLLECTION", Data_Collection),
        ("FEATURE EXTRACTION", Feature_Extraction),
        ("DISTILLATION ATTENTION", Distillation_Attention),
        ("IMAGE & TEXT FUSION", Img_Txt_Fusion),
        ("CLASSIFICATION", Classification),
        ("TRAINING", Training),
        ("PERFORMANCE METRICS", Performance_Metrics)
    ]
    for text, command in buttons:
        button = customtkinter.CTkButton(button_frame, text=text, command=command, **button_params)
        button.pack(pady=13)
    window.bind("<Configure>", bg_resizer)
    window.mainloop()

if __name__ == "__main__":
    main()