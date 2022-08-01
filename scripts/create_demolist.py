# Create a list of demo_images and demo_labels
import pandas as pd
def create_demolist(filepath):
    df = pd.read_csv(filepath)
    images = df.iloc[171:,0].to_list()
    images = ['data/'+i+'.npy' for i in images]
    labels = df.iloc[171:,10].to_list()

    return(images,labels)