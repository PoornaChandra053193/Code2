import os
import numpy as np
import mplcyberpunk
from qbstyles import mpl_style
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.family'] = 'Georgia'
os.makedirs("Graphs" , exist_ok = True)

def PerformanceMetrics():
    def AY():  # dark
        sym1 = '='*41
        print(f"\n{sym1}\n  1. Number of Epochs Vs. Accuracy Graph\n{sym1}\n")
        acc_list = np.loadtxt('ac.txt')
        num_Nodes_list = list(range(1, len(acc_list) + 1))
        plt.style.use('cyberpunk')
        plt.plot(num_Nodes_list, acc_list, color='#fcff68', marker='s', mec='#2eff73', mfc="#fd3535", linestyle='-.', linewidth=2, label="Proposed => Privacy-Preserving Federated Learning for Medical Visual Question Answering in Healthcare Systems")
        plt.title("Number of Epochs Vs. Accuracy Graph")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Accuracy (%)")
        plt.grid(True)
        plt.legend()
        mplcyberpunk.add_glow_effects(gradient_fill=True)
        plt.savefig("Graphs/Number of Epochs Vs. Accuracy.png")
        plt.show()
    def PN():# light
        sym2 = '='*42
        print(f"\n{sym2}\n  2. Number of Epochs Vs. Precision Graph\n{sym2}\n")
        pn_list = np.loadtxt('pn.txt')
        num_Nodes_list = list(range(1, len(pn_list) + 1))
        plt.style.use('Assets/Light.mplstyle')
        plt.plot(num_Nodes_list, pn_list, color='#469100', marker='o',markersize = 8,mec='#5653ff',mfc = "#ffa31a", linestyle='-.', linewidth=2,label='Proposed => Privacy-Preserving Federated Learning for Medical Visual Question Answering in Healthcare Systems')
        plt.title("Number of Epochs Vs. Precision Graph")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Precision (%)")
        plt.grid(True, color='#c2a500', linewidth=0.2)
        plt.legend()
        mplcyberpunk.add_glow_effects(gradient_fill=True)
        plt.savefig("Graphs/Number of Epochs Vs. Precision.png")
        plt.show()
    def FS():# dark
        sym3 = '='*42
        print(f"\n{sym3}\n  3. Number of Epochs Vs. F1-Score Graph\n{sym3}\n")
        fs_list = np.loadtxt('fs.txt')   
        num_Nodes_list = list(range(1, len(fs_list) + 1))
        mpl_style(dark=True)
        plt.plot(num_Nodes_list, fs_list, color='#64fff7', marker='s', mfc='#ff0000', mec="#fffc58", linestyle='-.', linewidth=2, label='Proposed => Privacy-Preserving Federated Learning for Medical Visual Question Answering in Healthcare Systems')
        plt.title("Number of Epochs Vs. F1-Score Rate Graph")
        plt.xlabel("Number of Epochs")
        plt.ylabel("F1-Score (%)")
        plt.grid(True, color='#cf7dff', linewidth=0.95)
        plt.legend()
        mplcyberpunk.add_glow_effects(gradient_fill=True)
        plt.savefig("Graphs/Number of Epochs Vs. F1-Score Rate.png")
        plt.show()    
    def RL():  # dark
        sym4 = '='*42
        print(f"\n{sym4}\n  4. Number of Epochs Vs. Recall Graph\n{sym4}\n")
        rl_list = np.loadtxt('rl.txt')
        num_Nodes_list = list(range(1, len(rl_list) + 1))
        plt.style.use('cyberpunk')
        bars = plt.bar(num_Nodes_list, rl_list, color='#a700d1', linestyle='-.', linewidth=2, label='Proposed => Privacy-Preserving Federated Learning for Medical Visual Question Answering in Healthcare Systems')
        plt.plot(num_Nodes_list, rl_list, color='#ffffff', marker='s', mfc='#ffee00', mec="#ff5858", linestyle='-.', linewidth=2)
        plt.title("Number of Epochs Vs. Recall Graph")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Recall (%)")
        plt.grid(True)
        plt.legend()
        mplcyberpunk.add_bar_gradient(bars=bars)
        mplcyberpunk.add_glow_effects(gradient_fill=True)
        plt.savefig("Graphs/Number of Epochs Vs. Recall.png")
        plt.show()

    def LS():  # dark
        sym1 = '='*41
        print(f"\n{sym1}\n  5. Number of Epochs Vs. Loss Graph\n{sym1}\n")
        ls_list = np.loadtxt('ls.txt')
        num_Nodes_list = list(range(1, len(ls_list) + 1))
        plt.style.use('cyberpunk')
        plt.plot(num_Nodes_list, ls_list, color='#8bff68', marker='s', mec='#fffc2e', mfc="#fd3535", linestyle='-.', linewidth=2, label="Proposed => Privacy-Preserving Federated Learning for Medical Visual Question Answering in Healthcare Systems")
        plt.title("Number of Epochs Vs. Loss Graph")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Loss (%)")
        plt.grid(True)
        plt.legend()
        mplcyberpunk.add_glow_effects(gradient_fill=True)
        plt.savefig("Graphs/Number of Epochs Vs. Loss.png")
        plt.show()
    def FNR():# light
        sym2 = '='*52
        print(f"\n{sym2}\n  6. Number of Epochs Vs. False Negative Rate Graph\n{sym2}\n")
        tpr_list = np.loadtxt('fnr.txt')
        num_Nodes_list = list(range(1, len(tpr_list) + 1))
        plt.style.use('Assets/Light.mplstyle')
        plt.plot(num_Nodes_list, tpr_list, color='#d80000', marker='o',markersize = 8,mec='#5653ff',mfc = "#1affd9", linestyle='-.', linewidth=2,label='Proposed => Privacy-Preserving Federated Learning for Medical Visual Question Answering in Healthcare Systems')
        plt.title("Number of Epochs Vs. False Negative Rate Graph")
        plt.xlabel("Number of Epochs")
        plt.ylabel("False Negative Rate (%)")
        plt.grid(True, color='#c2a500', linewidth=0.2)
        plt.legend()
        mplcyberpunk.add_glow_effects(gradient_fill=True)
        plt.savefig("Graphs/Number of Epochs Vs. False Negative Rate.png")
        plt.show()
    def FPR():# dark
        sym3 = '='*51
        print(f"\n{sym3}\n  7. Number of Epochs Vs. False Positive Rate Graph\n{sym3}\n")
        fpr_list = np.loadtxt('fpr.txt')   
        num_Nodes_list = list(range(1, len(fpr_list) + 1))
        mpl_style(dark=True)
        plt.plot(num_Nodes_list, fpr_list, color='#64ffbe', marker='s', mfc='#ff0000', mec="#fffc58", linestyle='-.', linewidth=2, label='Proposed => Privacy-Preserving Federated Learning for Medical Visual Question Answering in Healthcare Systems')
        plt.title("Number of Epochs Vs. False Positive Rate Graph")
        plt.xlabel("Number of Epochs")
        plt.ylabel("False Positive Rate (%)")
        plt.grid(True, color='#cf7dff', linewidth=0.95)
        plt.legend()
        mplcyberpunk.add_glow_effects(gradient_fill=True)
        plt.savefig("Graphs/Number of Epochs Vs. False Positive Rate.png")
        plt.show()    

    AY()#dark
    PN()#light
    FS()#light
    RL()#dark
    LS()#dark
    FNR()#light
    FPR()#dark
    os.remove(["ac.txt","pn.txt","rl.txt","fs.txt","ls.txt","fnr.txt","fpr.txt",".metric.json"])