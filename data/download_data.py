import os
import urllib.request

train_url = "https://raw.githubusercontent.com/ankita1112/House-Prices-Advanced-Regression/master/train.csv"
test_url = "https://raw.githubusercontent.com/ankita1112/House-Prices-Advanced-Regression/master/test.csv"

# Resolve path relative to script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = SCRIPT_DIR

os.makedirs(data_dir, exist_ok=True)

try:
    print("Downloading train.csv...")
    urllib.request.urlretrieve(train_url, os.path.join(data_dir, "train.csv"))
    print("Downloading test.csv...")
    urllib.request.urlretrieve(test_url, os.path.join(data_dir, "test.csv"))
    print("Download completed successfully.")
except Exception as e:
    print(f"Error downloading data: {e}")
