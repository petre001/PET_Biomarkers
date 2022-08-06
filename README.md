# PET_Biomarkers
## Repository for AIPI540 project

### Category: Health, wellness and fitness
### Type: CNN's
### Background & Situation
Amyloid status is important in evaluating mild cognitive impairment (MCI) patients; however, PET amyloid scans are expensive and involve ionizing radiation. This repo contains a deep learning algorithm that predicts amyloid status from MRI scans, an already existing component of the medical workup for MCI.

### Links to original datasets

https://adni.loni.usc.edu

### Links to papers

Tosun D, Veitch D, Aisen P, Jack CR, Jr., Jagust WJ, Petersen RC, et al. Detection of β-amyloid positivity in Alzheimer’s Disease Neuroimaging Initiative participants with demographics, cognition, MRI and plasma biomarkers. Brain Communications. 2021;3(2):fcab008. 

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8023542/

### Project Structure
main.py: main script for running demo

data: folder with csv file with names of image files and labels in 192 subjects with MCI; also contains .npy 3D image files for demo

scripts: folder with scripts for creating a list of demo images and labels and visualizing results

models: contains trained model for amyloid status prediction with input the images in data folder and output

### Instructions to run Streamlit app.
Use Streamlit to predict amyloid status from MRI scans in 21 demo subjects, in batches of 4.

#### Install dependencies from requirements.txt file
pip install -r requirements.txt

#### Start the Streamlit app
streamlit run main.py

