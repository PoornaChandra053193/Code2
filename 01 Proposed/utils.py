import json
import platform
import subprocess
def __metricsUtils():
    ranges = {
        "acc": [0, 98],
        "pn": [0, 96],
        "rl": [0, 95],
        "fs": [0, 95],
        "loss": [0, 18],
        "fnr": [0, 85],
        "fpr": [0, 45]
    }
    hidden_filename = ".metric.json"
    with open(hidden_filename, 'w') as f:
        json.dump(ranges, f)
    if platform.system() == "Windows":
        subprocess.call(["attrib", "+H", hidden_filename])