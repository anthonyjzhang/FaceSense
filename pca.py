import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# load the data
faces = pd.read_csv('faces.txt', sep = ' ', header = None)
faces = faces[0].str.split(',', expand=True)
faces = faces.astype('int')

# exchage the columns with the rows 
faces = faces.T

# add all index by 1
faces.index = faces.index + 1
faces

# To conduct PCA, we need to determine all the eigenvalues and eigenvectors

# First, center the data
faces = faces - faces.mean()

# Calculate the covariance matrix
cov = np.cov(faces)

# Calculate the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov)

# Sort the eigenvalues
eigenvalues = np.sort(eigenvalues)[::-1]

# Display Eigenvalues
plt.figure(figsize=(20, 10))
plt.bar(np.arange(1, 101), eigenvalues[:100])
plt.title('Top 150 Eigenvalues')
plt.xlabel('Index')
plt.ylabel('Eigenvalues')
plt.show()

# Conduct PCA using sklearn
from sklearn.decomposition import PCA
# we want to keep 90% of the variance
pca = PCA(n_components = 2)
pca.fit(faces)
faces_pca = pca.transform(faces)

# Plot the data
# Every ten points is a face, label with different combination of colors and markers
for i in range(0, 400, 10):
    plt.scatter(faces_pca[i:i+10, 0], faces_pca[i:i+10, 1], label = 'Face {}'.format(i//10))
plt.title('PCA of Faces')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# Dispay raw data
faces_pca = pd.DataFrame(faces_pca)
faces_pca.index = faces.index
faces_pca

# Show face
face1 = faces.iloc[0,:]
face1 = face1.values.reshape(92,112)
face1 = face1.T
plt.imshow(face1, cmap = 'gray')
plt.show()

# Principle components
ratios = [0.6, 0.7, 0.8, 0.85, 0.9]
for ratio in ratios:
    pca = PCA(n_components = ratio)
    pca.fit(faces)
    faces_pca = pca.transform(faces)
    faces_pca = pd.DataFrame(faces_pca)
    faces_pca.index = faces.index
    r = int(pca.explained_variance_ratio_.sum()*100)
    print('The number of principal components is {} and the ratio of variance is {}%'.format(pca.n_components_, r))
    faces_pca.to_csv('pca_{}%.csv'.format(r), header = None, index = None)

pca = PCA()
pca.fit(faces)
faces_pca = pca.transform(faces)
faces_pca = pd.DataFrame(faces_pca)
faces_pca.index = faces.index
faces_pca.to_csv('pca_100%.csv', header = None, index = None)

cumsum = np.cumsum(pca.explained_variance_ratio_)

# Plot cunulative sum of explained variance ratio
plt.plot(cumsum)

# Scatter the points when the ratio is in the ratios list
for ratio in ratios:
    plt.scatter(np.where(cumsum >= ratio)[0][0], ratio, color = 'red')
plt.title('Cumulative Sum of Explained Variance Ratio')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Sum of Explained Variance Ratio')
plt.show()
