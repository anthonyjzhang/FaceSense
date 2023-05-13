import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def concat_data(filename):
    faces = pd.read_csv(filename,header=None)

    # load the data
    nonface = pd.read_csv('not_faces.txt', sep = ' ', header = None)
    nonface = nonface[0].str.split(',', expand=True)
    nonface = nonface.astype('float')
    # exchage the columns with the rows 
    nonface = nonface.T
    # add all index by 1
    nonface.index = nonface.index + 1
    # use PCA to reduce nonface to the same dimension as faces
    from sklearn.decomposition import PCA
    n_components = faces.shape[1]
    pca = PCA(n_components=n_components)
    pca.fit(nonface)
    nonface = pca.transform(nonface)
    nonface = pd.DataFrame(nonface)

    # Pick up first 90 rows of faces, label the first 10 as 1, then 2, and so on
    faces = faces.iloc[0:90,:]
    faces['label'] = 10*[1]+10*[2]+10*[3]+10*[4]+10*[5]+10*[6]+10*[7]+10*[8]+10*[9]
    # Label the nonface data as 0
    nonface['label'] = 400*[0]
    
    # Concatenate the faces and nonface data
    data = pd.concat([faces, nonface], ignore_index=True)
    return data

filename = ['pca_61%.csv','pca_70%.csv', 'pca_80%.csv','pca_85%.csv','pca_90%.csv','pca_100%.csv']
for i in filename:
    data = concat_data(i)
    # use RE to get the number before %
    import re
    num = re.findall(r"\d+\.?\d*",i)
    data.to_csv('clf_pca'+num[0]+'%.csv',index=False,header=False)

faces = pd.read_csv('faces.txt', sep = ' ', header = None)
faces = faces[0].str.split(',', expand=True)
faces = faces.astype('int')
# exchage the columns with the rows 
faces = faces.T
# add all index by 1
faces.index = faces.index + 1

nonface = pd.read_csv('not_faces.txt', sep = ' ', header = None)
nonface = nonface[0].str.split(',', expand=True)
nonface = nonface.astype('float')
# exchage the columns with the rows 
nonface = nonface.T
# add all index by 1
nonface.index = nonface.index + 1

faces = faces.iloc[0:90,:]
faces['label'] = 10*[1]+10*[2]+10*[3]+10*[4]+10*[5]+10*[6]+10*[7]+10*[8]+10*[9]
# label the nonface data as 0
nonface['label'] = 400*[0]

# concat the faces and nonface data
data = pd.concat([faces, nonface], ignore_index=True)
data.to_csv('clf_raw.csv',index=False,header=False)