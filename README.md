# Description

### Step 1: Download the dataset and pre-trained models
``

### Step 2: Extract archived info
``

### Verify
# TODO
directory tree should be here

### How to run evaluation?
##### Requirements
- Python version `3.12.0`
- Conda environment \
Tip: `conda create -n squire python=3.12.0` to create a conda environment named `squire` with Python 3.12.0
- Install torch version `v2.7.1` from [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/)
- Install the required packages in the conda environment `pip install -r requirements.txt` to install the required packages in the conda environment.

##### Run evaluation

We provide multiple example files of bash shell, such as:
- `run_metaqa.sh.example`
- `run_mquake.sh.example`
- `run_kinship.sh.example`

Please copy those files and remove the `.example` suffix, and then edit the variables in those files to point to the correct paths. After that, you can run those bash shell files to execute the whole pipeline of training and evaluation. \
Run `cp run_metaqa.sh.example run_metaqa.sh` and `cp run_mquake.sh.example run_mquake.sh` and `cp run_kinship.sh.example run_kinship.sh` to copy the example files.

Inside of those bash shell files: \
- Change `ROOT=path/SQUIRE` to the correct path of your SQUIRE repository.
- In order to test with/without paraphrased questions, `USE_PARAPHRASED=true` or `USE_PARAPHRASED=false` in those bash shell files 
- Change `CONDA_ENV=name_env` to the name of your conda environment where SQUIRE is installed. 
- (Only for Mquake) Change `dataType="single/multi"` to either "single" or "multi" depending on whether you want to evaluate on MQuAKE-Single Answer or MQuAKE-Multi Answer dataset.
 
Execute `bash run_metaqa.sh eval` to run evaluation for MetaQA dataset. \
Execute `bash run_mquake.sh eval` to run evaluation for MQuAKE dataset. \
Execute `bash run_kinship.sh eval` to run evaluation for Kinship dataset
