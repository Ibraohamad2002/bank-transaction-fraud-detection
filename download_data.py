import subprocess
import os

os.makedirs("data", exist_ok=True)

subprocess.run([
    "python", "-m", "kaggle",
    "datasets", "download",
    "-d", "nafiulislam490/bank-transaction-fraud-detection-dataset",
    "-p", "data"
], check=True)

print("Dataset downloaded successfully!")