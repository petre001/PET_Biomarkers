def visualize_results(model, images, labels, batch_size, device):
    # Display a batch of predictions
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import streamlit as st

    model = model.to(device)  # Send model to GPU if available
    batch_size = batch_size

    # Set up a mapping dictionary
    classes = ['A-', 'A+']
    idx_to_class = {i: j for i, j in enumerate(classes)}
    class_to_idx = {v: k for k, v in idx_to_class.items()}

    with torch.no_grad():
        model.eval()
        # Get a batch of demo images
        #images, labels = next(iter(dataloader))
        images, labels = images.to(device), labels.to(device)
        # Get predictions
        _, preds = torch.max(model(images), 1)
        preds = np.squeeze(preds.cpu().numpy())
        labels = labels.cpu().numpy()
        images = images.cpu().numpy()  # Convert images to numpy for display

    #print(images[:, 0, 40, :, :][0].shape)
    print(preds)

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(batch_size/2, batch_size/1.5))
    for idx in range(batch_size):
        ax = fig.add_subplot(int(batch_size/2), 2, idx + 1, xticks=[], yticks=[])
        # ax.imshow(np.rot90(im[:,0,40,:,:][idx]), cmap='gray')
        ax.imshow(images[:, 0, :, :, 47][idx], cmap='gray')
        # ax.set_title(idx_to_class[label[idx]])
        ax.set_title("{} ({})".format(idx_to_class[preds[idx]], idx_to_class[labels[idx]]),
                     color=("green" if preds[idx] == labels[idx] else "red"))
    #plt.show()
    st.pyplot(fig=fig)
    return
