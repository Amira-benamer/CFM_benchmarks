#!/usr/bin/env python3
"""
Minimal cost tracking script for EC2 spot instances.
Logs estimated cost to /home/ubuntu/cost_log.txt every 5 minutes.
"""
import time
import boto3
import os
from datetime import datetime

INSTANCE_ID_URL = "http://169.254.169.254/latest/meta-data/instance-id"
LOG_FILE = "/home/ubuntu/cost_log.txt"
INTERVAL = 300  # 5 minutes

# Spot price mapping (update as needed)
PRICING = {
    "t3.medium": 0.0456,
        "g4dn.xlarge": 0.587,  # 1x T4 16GB
      "p4d.24xlarge": 32.77,  # 8x A100 40GB
      "p3.2xlarge": 3.06,  # 1x V100 16GB
      "g5.xlarge": 1.01,  # 1x A10G 24GB
      "g5.4xlarge": 4.03,  # 1x A10G 24GB
}

def get_instance_id():
    try:
        import requests
        return requests.get(INSTANCE_ID_URL, timeout=2).text
    except Exception:
        return os.popen(f"curl -s {INSTANCE_ID_URL}").read().strip()

def get_instance_type():
    try:
        return os.popen("curl -s http://169.254.169.254/latest/meta-data/instance-type").read().strip()
    except Exception:
        return "unknown"

import atexit

def main():
    instance_id = get_instance_id()
    instance_type = get_instance_type()
    hourly_rate = PRICING.get(instance_type, 1.0)
    start_time = time.time()

    def log_cost(event="periodic"):
        elapsed_hours = (time.time() - start_time) / 3600
        est_cost = elapsed_hours * hourly_rate
        with open(LOG_FILE, "a") as f:
            f.write(f"[{datetime.now().isoformat()}] {event} cost log: Elapsed: {elapsed_hours:.2f}h, Estimated cost: ${est_cost:.2f}\n")

    # Log initial cost at startup
    with open(LOG_FILE, "a") as f:
        f.write(f"Cost tracking started for {instance_id} ({instance_type}) at {datetime.now().isoformat()}\n")
    log_cost(event="initial")

    # Register shutdown handler to log final cost
    atexit.register(lambda: log_cost(event="final"))

    while True:
        log_cost(event="periodic")
        time.sleep(INTERVAL)

if __name__ == "__main__":
    main()

