import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# load the data
faces = pd.read_csv('faces.txt', sep = ' ', header = None)
faces = faces[0].str.split(',', expand=True)
faces = faces.astype('float')

# exchage the columns with the rows 
faces = faces.T

# add all index by 1
faces.index = faces.index + 1

# Show csv data
print(faces)

# create a for loop to plot all the faces in ten different figures
for j in range(40):
    plt.figure(figsize = (10, 5))
    for i in range(10):
        
        face = faces.iloc[10*j+i,:]
        face = face.values.reshape(92,112)
        face = face.T
        plt.subplot(2, 5, i+1)
        plt.imshow(face, cmap = 'gray')
        plt.axis('off')
    plt.show()