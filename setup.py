# -*- coding: utf-8 -*-

# Predicting PET Biomarkers of Alzheimer’s Disease With MRI Using Deep Convolutional Neural Networks 
### Contributors: Jeffrey Petrella

# This project uses transfer learning to train a ResNet18 model to identify amyloid PET biomarker status from MRI images. It should be run on a GPU

### Step 1: Assume we're in the GitHub repo

"""### Step 2: Install and Import Dependencies"""

# Install dependencies from requirements.txt file
#pip install -r requirements.txt

# Commented out IPython magic to ensure Python compatibility.
#git clone https://github.com/Project-MONAI/MONAI.git
# %cd MONAI/
#pip install -e '.[all]'

import os
import pandas as pd
import torch
import monai
from monai.data import DataLoader, ImageDataset, NumpyReader
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, EnsureType

pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

"""### Step 3: Load Training and Test data"""

#from google.colab import drive
#drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/test_mci_data

# Create a list of images and labels
df = pd.read_csv('data/MCI_labels.csv')
images = df.iloc[:,0].to_list()
images = [i+'.npy' for i in images]
labels = df.iloc[:,10].to_list()

# Define transforms
train_transforms = Compose([ScaleIntensity(), AddChannel(), Resize((96, 96, 96)), RandRotate90(), EnsureType()])
val_transforms = Compose([ScaleIntensity(), AddChannel(), Resize((96, 96, 96)), EnsureType()])

# Create training Dataset and DataLoader using first 171 images
batch_size = 2

train_ds = ImageDataset(image_files=images[:170], labels=labels[:170], transform=train_transforms, reader='NumpyReader')
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

# Create validation Dataset and DataLoader using the rest of the 21 images
val_ds = ImageDataset(image_files=images[171:], labels=labels[171:], transform=val_transforms, reader='NumpyReader')
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

# Set up dict for dataloaders
dataloaders = {'train':train_loader,'val':val_loader}

# Store size of training, validation and test sets
dataset_sizes = {'train':len(train_ds),'val':len(val_ds)}

im, label = monai.utils.misc.first(train_loader)
print(f'Image type: {type(im)}')
print(f'Input batch shape: {im.shape}')
print(f'Label batch shape: {label.shape}')

# Set up a mapping dictionary
classes = ['Amyloid(-)','Amyloid(+)']
idx_to_class = {i:j for i,j in enumerate(classes)}
class_to_idx = {v:k for k,v in idx_to_class.items()}

"""###Step 4: Define our model architecture
We will used a pre-trained DenseNet 121 model for this task.
"""

# Load a pre-trained DenseNet121
# We have a signle input channel, and we have 2 output classes
# We set spatial_dims=3 to indicate we want to use the version suitable for 3D input images
model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)

"""### Step 5: Train the Model"""

def train_model(model, criterion, optimizer, dataloaders, device, num_epochs=5):

    model = model.to(device) # Send model to GPU if available

    iter_num = {'train':0,'val':0} # Track total number of iterations

    best_metric = -1

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Get the input images and labels, and send to GPU if available
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the weight gradients
                optimizer.zero_grad()

                # Forward pass to get outputs and calculate loss
                # Track gradient only for training data
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backpropagation to get the gradients with respect to each weight
                    # Only if in train
                    if phase == 'train':
                        loss.backward()
                        # Update the weights
                        optimizer.step()

                # Convert loss into a scalar and add it to running_loss
                running_loss += loss.item() * inputs.size(0)
                # Track number of correct predictions
                running_corrects += torch.sum(preds == labels.data)

                # Iterate count of iterations
                iter_num[phase] += 1

            # Calculate and display average loss and accuracy for the epoch
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Save weights if accuracy is best
            if phase=='val':
                if epoch_acc > best_metric:
                    best_metric = epoch_acc
                    if not os.path.exists('./models'):
                        os.mkdir('./models')
                    torch.save(model.state_dict(),'models/3d_classification_model.pth')
                    print('Saved best new model')

    print(f'Training complete. Best validation set accuracy was {best_metric}')
    
    return

# Use cross-entropy loss function
criterion = torch.nn.CrossEntropyLoss()
# loss_function = torch.nn.BCEWithLogitsLoss()  # also works with this data

# Use Adam adaptive optimizer
optimizer = torch.optim.Adam(model.parameters(), 1e-4)

# Train the model
epochs=5
train_model(model, criterion, optimizer, dataloaders, device, num_epochs=epochs)