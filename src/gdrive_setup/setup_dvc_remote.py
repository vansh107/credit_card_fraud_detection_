import json
import subprocess

# Path to the gdrive.json file
gdrive_json_path = "config/gdrive.json"

# Load the client_id and client_secret from the JSON file
with open(gdrive_json_path, "r") as f:
    data = json.load(f)
    client_id = data["installed"]["client_id"]
    client_secret = data["installed"]["client_secret"]

# Configure DVC remote with client_id and client_secret
try:
    subprocess.check_call(["dvc", "remote", "modify", "myremote", "gdrive_client_id", client_id])
    subprocess.check_call(["dvc", "remote", "modify", "myremote", "gdrive_client_secret", client_secret])
    print("DVC remote configured successfully with client ID and client secret.")
except subprocess.CalledProcessError as e:
    print("An error occurred while configuring the DVC remote:", e)
