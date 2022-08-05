import monai
import torch
from monai.data import DataLoader, ImageDataset
from monai.transforms import AddChannel, Compose, Resize, ScaleIntensity, EnsureType
import streamlit as st
from scripts.create_demolist import create_demolist
from scripts.make_predictions import visualize_results

pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.title('Prediction of Amyloid Status from MRI Scans')
st.header('Load a new set of images')

'Starting computation of new predictions'

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)
latest_iteration.text('Progress')

# load demo image names and labels
images, labels = create_demolist('data/MCI_labels.csv')
bar.progress(25)

# Load model
model2 = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
model2.load_state_dict(torch.load("models/3d_classification_model.pth", map_location=torch.device('cpu')))
bar.progress(50)

# Define transforms
demo_transforms = Compose([ScaleIntensity(), AddChannel(), Resize((96, 96, 96)), EnsureType()])

# Create demo Dataset and DataLoader
batch_size = 4
demo_ds = ImageDataset(image_files=images, labels=labels, transform=demo_transforms, reader='NumpyReader')
demo_loader = DataLoader(demo_ds, batch_size=batch_size, shuffle=True, num_workers=0,
                         pin_memory=torch.cuda.is_available())
bar.progress(75)
# Create Iterator object
data_iter = iter(demo_loader)

result = st.button('Load')
if result:
    # Iterate the dataloader
    images, labels = next(data_iter)

    # Make predictions and visualize results
    visualize_results(model2, images, labels, batch_size, device)
    bar.progress(100)


