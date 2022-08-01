from multiprocessing import freeze_support

import torch
import monai
from monai.data import DataLoader, ImageDataset, NumpyReader
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, EnsureType
from scripts.create_demolist import create_demolist
from scripts.make_predictions import visualize_results

pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load demo image names and labels
images, labels = create_demolist('data/MCI_labels.csv')

# Load model
model2 = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
model2.load_state_dict(torch.load("models/3d_classification_model.pth", map_location=torch.device('cpu')))

# Define transforms
demo_transforms = Compose([ScaleIntensity(), AddChannel(), Resize((96, 96, 96)), EnsureType()])

# Create demo Dataset and DataLoader
batch_size = 8
demo_ds = ImageDataset(image_files=images, labels=labels, transform=demo_transforms, reader='NumpyReader')
demo_loader = DataLoader(demo_ds, batch_size=batch_size, shuffle=True, num_workers=0,
                         pin_memory=torch.cuda.is_available())

# Make predictions and visualize results
visualize_results(model2, demo_loader, batch_size, device)


