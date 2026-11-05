import os
import subprocess

os.makedirs("data", exist_ok=True)

subprocess.run(
    [
        "python",
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        "kartik2112/fraud-detection",
        "-p",
        "data",
        "--unzip",
    ],
    check=True,
)

print("Dataset downloaded successfully!")
