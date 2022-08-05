# PET_Biomarkers
## Repository for AIPI540 project

### Category: Health, wellness and fitness
### Type: CNN's
### Background & Situation
Amyloid status is important in evaluating MCI patients; however, PET amyloid scans are expensive and involve ionizing radiation. This repo contains a deep learning algorithm that predicts amyloid status from MRI scans, an already existing component of the medical workup for MCI.

### Links to original datasets

Because of the size of the original dataset, it is not in a publically accessible repo.

### Links to papers


### Project Structure
main.py: main script for running demo
model.py
data: folder with csv. File with names of image files and labels in 192 subjects with MCI
scripts: folder with scripts for creating a list of demo images and labels and visualizing results

Instructions to run Streamlit app.
Use Streamlit to predict amyloid status from MRI scans in 21 demo subjects, in batches of 4.

##Install the requirements needed to use Streamlit
### Install dependencies from requirements.txt file
pip install -r requirements.txt

### Install MONAI tools
git clone https://github.com/Project-MONAI/MONAI.git
%cd MONAI/
pip install -e '.[all]'

###Start the Streamlit app
make run

