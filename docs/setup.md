python -m venv .mlops_venv
source .mlops_venv/bin/activate

pip install -e .

dvc init

dvc dag

dvc repro

dvc remote add --default -f myremote gdrive://path to g-drive folder/dvcstore

python3 src/gdrive_setup/setup_dvc_remote.py 

git push
