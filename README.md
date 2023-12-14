# MACHINE-LEARNING-INTRUSION-DETECTION-SYSTEM
ML project based on intrusion detection system trained dataset
# Intrusion Detection System Using Machine Learning Algorithms
# Intrusion Detection System Review 2
## Project Overview

Welcome to our Intrusion Detection System (IDS) project, where we leverage machine learning algorithms to enhance network security. Our team has made significant contributions to various aspects of the project, ranging from data preprocessing and encoding to model analysis and visualization.


## Table of Contents

1. **Pre-processing**
   - Description: Transforming raw data into a clean dataset.
   - Steps:
     - Loading and verifying all dataset attributes.
     - Summary statistics: mean, median, mode, etc.
     - Checking for missing data (no missing values in this project).
     - Removing duplicate rows.
     - Label Encoding: Converting categorical variables to numerical.

2. **Feature Selection**
   - Heatmaps to understand attribute correlations.
   - Recursive Feature Elimination (RFE) using Random Forest Classifier and Decision Tree Classifier.
   - Selected Features: Comparison between CLF and RFE.

3. **Feature Engineering**
   - Manual engineering with domain ideology.
   - Creation of new attributes using logical conditions.
   - Binning technique for better handling of continuous numeric features.
   - Log transformation to mitigate the impact of outliers.

4. **Model Analysis**
   - Utilizing machine learning algorithms to learn patterns in datasets.
   - Graph plotting for attributes comparison before and after data cleaning.
   - Efficiency comparison after data cleaning.

5. **Visualization**
   - Importance of data visualization.
   - Visualization after logistic regression, decision tree, and random forest classification.
   - Use of "SELECTKBEST" and "F_CLASSICF" for logistic regression.

6. **Hyperparameter Tuning**
   - Using GridSearchCV for an exhaustive search through hyperparameter space.

7. **Error/Loss Curves**
   - Importance of loss curves in understanding model performance.
   - Learning curves for logistic regression and accuracy curves for decision tree.

8. **Performance Metrics**
   - Evaluation metrics using classification report and accuracy score.
   - Metrics for Logistic Regression, Decision Tree, Naïve Bayes, SVM.

## Resources

- [GeeksforGeeks - Intrusion Detection System using Machine Learning Algorithms](https://www.geeksforgeeks.org/intrusion-detection-system-using-machine-learning-algorithms/)
- [Palo Alto Networks - What is an Intrusion Detection System (IDS)](https://www.paloaltonetworks.com/cyberpedia/what-is-an-intrusion-detection-system-ids)
- [Journal of Big Data - Machine learning for intrusion detection in IoT big data](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-018-0145-4)
- [ITM Web of Conferences - Intrusion Detection System using Machine Learning and Data Mining Techniques](https://www.itmconferences.org/articles/itmconf/abs/2022/06/itmconf_iceas2022_02003/itmconf_iceas2022_02003.html)
- [Kaggle Datasets](https://www.kaggle.com/datasets)

## Conclusion

We have successfully applied machine learning algorithms to create an efficient Intrusion Detection System. Our evaluation includes preprocessing, feature engineering, and model analysis, laying the foundation for extending the project with other algorithms.

**Thanks for exploring our IDS project!**

*Note: This README is a template and should be adapted to fit the specific details and structure of your project.*
# Intrusion Detection System Review 2

## Introduction

Welcome to the second review of our Intrusion Detection System (IDS) project! In this review, we delve into various machine learning models employed for intrusion detection. The collaboration and contributions of team members are highlighted for transparency and acknowledgment.


## Why Machine Learning in IDS?

Machine Learning in Intrusion Detection Systems enhances security by learning normal patterns and detecting anomalies. It adapts to evolving threats, reducing false positives through behavioral analysis, enabling efficient identification of complex attack patterns.

## Our Models in Review 2

### 1. **Probabilistic Model**
   - **Gaussian Mixture Model (GMM)**
   - **Naive Bayes Classifier Model**
   - **Hidden Markov Model (HMM)**

### 2. **Unsupervised Model**
   - **K-Means Clustering Model**
   - **DB-SCAN Model**
   - **Hierarchical Clustering**

### 3. **Dimensionality Reduction**
   - **PCA (Principal Component Analysis)**
   - **t-SNE (t-distributed Stochastic Neighbour Embedding) Model**

### 4. **Ensemble Model**
   - **Random Forest**
   - **Gradient Boosting Model**

## Model Details and Visualization

### 1. Probabilistic Model

#### Gaussian Mixture Model (GMM)
- GMM is a statistical model expressing a dataset as a mix of Gaussian distributions.
- Used for tasks like clustering and density estimation.

#### Naive Bayes Classifier Model
- A probabilistic model based on Bayes' theorem, efficient for text classification and spam filtering.

#### Hidden Markov Model (HMM)
- Statistical models representing a system with hidden states, widely used in speech recognition and natural language processing.

*Visualizations, loss curves, and classification reports are provided for each model.*

**Statistics:**
- Naive Bayes exhibited significantly higher accuracy compared to GMM and HMM.

### 2. Unsupervised Model

#### K-Means Clustering Model
- A partitioning algorithm grouping data points based on feature similarities to minimize within-cluster variance.

#### DB-SCAN Model
- Density-based clustering algorithm identifying clusters as dense regions separated by sparser areas.

#### Hierarchical Clustering
- Agglomerative or divisive approach organizing data into a tree-like hierarchy of clusters.

*Visualizations, loss curves, and comparisons are provided for each model.*

**Statistics:**
- K-Means showed significantly higher Silhouette Score compared to DBSCAN and Hierarchical Clustering.

### 3. Dimensionality Reduction

#### PCA (Principal Component Analysis)
- Technique transforming high-dimensional data into a lower-dimensional space while preserving variance.

#### t-SNE (t-Distributed Stochastic Neighbor Embedding) Model
- Dimensionality reduction technique visualizing high-dimensional data by preserving local similarities.

*Visualizations, loss curves, and comparisons are provided for each model.*

**Statistics:**
- PCA exhibited significantly higher accuracy compared to t-SNE.

### 4. Ensemble Model

#### Random Forest
- Ensemble learning method constructing decision trees during training for robust and accurate predictions.

#### Gradient Boosting Model
- Ensemble learning technique combining weak predictive models sequentially to create a strong predictive model.

*Visualizations, loss curves, and classification reports are provided for each model.*

**Statistics:**
- XGB Boosting exhibited significantly higher accuracy compared to Random Forest.

## Conclusion

After a comprehensive review and comparison, we conclude that for our IDS dataset, XGB Boosting is the perfect ensemble model, offering high accuracy compared to other models.

## Resources

- [GeeksforGeeks - Intrusion Detection System using Machine Learning Algorithms](https://www.geeksforgeeks.org/intrusion-detection-system-using-machine-learning-algorithms/)
- [Palo Alto Networks - What is an Intrusion Detection System (IDS)](https://www.paloaltonetworks.com/cyberpedia/what-is-an-intrusion-detection-system-ids)
- [Journal of Big Data - Machine learning for intrusion detection in IoT big data](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-018-0145-4)
- [ITM Web of Conferences - Intrusion Detection System using Machine Learning and Data Mining Techniques](https://www.itmconferences.org/articles/itmconf/abs/2022/06/itmconf_iceas2022_02003/itmconf_iceas2022_02003.html)

For additional datasets, you can visit [Kaggle Datasets](https://www.kaggle.com/datasets).

**Thank you for exploring our IDS project!**

*Note: This README is a template and should be adapted to fit the specific details and structure of your project.*
