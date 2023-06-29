# 👥 FaceSense

## Background
This project is based on the famous "Eigenfaces" paper, where a facial recognition system using Principal Component Analysis (PCA) algorithm was implemented. PCA is a unsupervised machine learning technique used for reducing the dimensionality of data. In the file titled 'faces.txt', there are 400 112x92 pixel greyscale images of 40 distinct people (10 images per person). The file titled 'nonfaces.txt' similarly includes 112x92 greyscale images, but of nonfaces (random objects). Using PCA, the features of this large dataset are reduced such that a small set of significant features are used to describe the variation between distinct facial images. After dimension reduction, the algorithm attempts to use supervised machine learning classification techniques like K-nearest Neighbors and Random Forests to classify an image as Unknown, Face, and Not a Face, as well as distinguish between separate faces, i.e. distinguish between Subject A, Subject B, Subject C, and so forth. An example of the facial and nonfacial images is shown below:

![image](https://github.com/anthonyjzhang/Facial-Recognition/assets/97823062/210240c9-6be8-4a38-8be4-125b78d29516)
![image](https://github.com/anthonyjzhang/Facial-Recognition/assets/97823062/4f24b517-bde6-44df-8df5-afb3f57eeba5)

## Program

### Results
In the Cumulative Sum of Explained Variance Ratio Graph shown below, the curve first rises rapidly and then gradually levels off. 

<img width="608" alt="Screen Shot 2023-05-13 at 5 17 11 AM" src="https://github.com/anthonyjzhang/Facial-Recognition/assets/97823062/af22d974-7ae2-4e5e-bf93-5497efc1e655">

This shows that at the beginning (when the number of principal components is small), increasing the number of principal components can greatly increase the explained variance ratio. When the number of principal components becomes large (100-150+), it is no longer meaningful to increase the number of principal components in exchange for a small increase in the explained variance ratio. Thus, a law of diminshing returns applies when reducing the dimensionality of the image data. In order to determine the optimal parameter range, I selected several thresholds of explained variance ratio, 61%, 70%, 80%, 90%, and 95%, respectively. For the classification section, I tried different supervised machine learning classification methods including logistic regression, K-nearest neighbor and random forest. The first classification method investigated was logistic regression. The obtained results showed that the average accuracy was not very high, and in contradiction to expectations, as the ratio of explained variance increased, the average accuracy decreased. One possible explanation for the failure of logistic regression is that as a linear model, logistic regression can only handle linearly separable problems, and performs poorly for nonlinear problems. When the data exhibits a nonlinear relationship, logistic regression may underfit or overfit. Additionally, logistic regression is sensitive to noise and outliers, thus this may have impacted the perforamnce of logistic regression as a classifier.  Then, when using the K-Nearest Neighbors classification method, the results showed that K-NN achieved relatively ideal results, which was predictable because K-NN can handle nonlinear relationships and is not sensitive to noise. The average accuracy obtained by random forest was not much different from K-NN, but I decided not to use a random forest classifier due to its algorithmic complexity. After completing the binary classification, the goal was mutliple classification, or distinguishing the 40 different subjects in the faces images. Due to the previously discussed results, I chose K-NN as the model for this classification. The results showed that for this multi-classification problem, as the retained variance ratio increased, the average accuracy rose. This implied that more information causes the alogirthm to more accurately classify faces, as exepcted.

Overall, the results were credible, as shown in the figure below, the results demonstrated a trend between accuracy and PCA variance percentages. 

<img width="518" alt="Screen Shot 2023-05-13 at 5 13 18 AM" src="https://github.com/anthonyjzhang/Facial-Recognition/assets/97823062/1d612d63-c713-48f9-9f7f-2fe95d473ab4">


The results were promising because a high degree of accuracy was achieved using the dataset with reduced dimensions when classifying faces and non-faces, as well as distinguishing between different people. In order to improve these results in the future, in addition to dimension reduction, feature selection methods could be explored prior to classification in order to determine the most informative features for classification.

### Dimension Reduction
In its original state, the amount of data that needs to be processed in each image is far too inefficient to run and could even be impossible for many common computers. Thus, the images will need to be simplified using a Principal Component Analysis (PCA), a dimension reduction technique. First, each vector is put into a larger matrix. By multiplying the matrix by its transpose, the covariance matrix is obtained. From the covariance matrix, the eigenvalues are found using basic linear algebra. The principal components are linear combinations of the original data. The components each represent the vectors that capture the most data or in other terms the highest percentage of variance. Each eigenvalue represents one principal component with 10,304 total components. However, there are only 400 non-zero values, thus there are 400 principal components. The number of principal components saved depends on the desired amount of variance to be retained. The goal was to obtain the highest percentage of variance with the fewest number of principal components in order to maximize the alogirhtm's result accuracy and while minimizing the algorithm's data processing power. 

### Classification
After PCA was conducted, various classification methods were investigated. The classification of the data was split into two parts: binary classification and multiple classification. In addition to the 400 pictures of faces, 90 additional pictures of “nonfaces” were added to the data set. The goal of binary classification was to distinguish between '"face images" and "nonface images"". Once again, the goal was to use a greater ratio of variance with fewer principal components, determining whether the pictures depicted "faces" or “nonfaces.” The K-Nearest Neighbors or KNN binary classification method was utilized. The data was first split into 5 stratified K-folds, splitting the folds into training data and test data. The training data was used to predict the values of the test data, and the accuracy and precision of the predicted values was then taken and analyzed to aid in a 10-NN classification of the data, predicting the value of a data point based on the 10 nearest labeled data values. It can be noted that another binary classification function was used – the random forest method, which uses a series of decision tree classifications to predict data values. The random forest method yielded similar results compared to the K-NN classification method. In addition to the binary classification, multiple classification was needed to distinguish between the 40 different people among the 400 facial images. The same process was used, implementing a 10-NN classification of the data and analyzing the accuracy using the PCA percentages. 

## Softwares and Technologies

<div align="center">
  <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54"/>
 <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white"/>
 <img src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white"/>
 <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
</div>



