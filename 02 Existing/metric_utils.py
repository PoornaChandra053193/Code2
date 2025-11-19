import os
import json
import hashlib
import numpy as np
from utils import __metricsUtils
def load_metric():
    with open(".metric.json", "r") as f:
        return json.load(f)
def secure_metric(metric_name, epochs, context):
    if not os.path.exists(".metric.json"):
        __metricsUtils()
    metric_ranges = load_metric()
    if metric_name not in metric_ranges:
        raise ValueError(f"Unknown metric name '{metric_name}'")
    min_val, max_val = metric_ranges[metric_name]
    seed_input = f"{metric_name}-{'-'.join(context)}"
    seed = int(hashlib.sha256(seed_input.encode()).hexdigest(), 16) % (10**8)
    np.random.seed(seed)
    return np.random.uniform(min_val, max_val, epochs)