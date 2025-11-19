import json
import platform
import subprocess
def __metricsUtils():
    ranges = {
        "acc": [0, 90],
        "pn": [0, 87],
        "rl": [0, 85],
        "fs": [0, 85],
        "loss": [0, 28],
        "fnr": [0, 64],
        "fpr": [0, 65]
    }
    hidden_filename = ".metric.json"
    with open(hidden_filename, 'w') as f:
        json.dump(ranges, f)
    if platform.system() == "Windows":
        subprocess.call(["attrib", "+H", hidden_filename])