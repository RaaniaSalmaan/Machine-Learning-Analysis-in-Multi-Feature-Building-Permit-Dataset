The main goal of this project is to apply and compare different Machine Learning techniques on a real-world dataset. The Building Permits dataset was used to perform classification, regression, and clustering tasks.
The project focuses on proper data preprocessing,model implementation, evaluation, and comparison of results to understand how different algorithms behave on complex and high-dimensional data.

-> Objectives
- Apply supervised learning for classification and regression
- Apply unsupervised learning to discover hidden patterns
- Handle categorical and numerical data effectively
- Reduce dimensionality using PCA and LDA
- Compare traditional ML models with advanced models

-> Dataset

- Dataset: Building Permits Dataset (Kaggle)
- Features: 35+ numerical and categorical attributes
- Targets:
  - Classification: `Current Status`
  - Regression: `Estimated Cost`

-> Machine Learning Techniques Used

 => Supervised Learning

**Classification Models:**

- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Naive Bayes
- Perceptron
- MLP Classifier
- CatBoost

**Regression Models:**

- Linear Regression
- Ridge Regression
- Lasso Regression
- MLP Regressor

-> Unsupervised Learning

- K-Means Clustering
- Agglomerative Hierarchical Clustering (HCA)
- Spectral Clustering

-> Dimensionality Reduction

- PCA (Principal Component Analysis): used for noise reduction, faster training, and visualization
- LDA (Linear Discriminant Analysis): used for supervised dimensionality reduction and class separation

-> Preprocessing Steps

 Removal of irrelevant and redundant columns
 Handling categorical features using:

  - One-Hot Encoding
  - Label / Ordinal Encoding
 Feature scaling using StandardScaler
 Dimensionality reduction using PCA and LDA

-> Evaluation Metrics

 **Classification:** Accuracy, F1-Score, Confusion Matrix
 **Regression:** R² Score, Mean Squared Error (MSE)
 **Clustering:** Silhouette Score
 **Validation:** K-Fold Cross Validation

-> Results Summary

- Supervised models achieved **very high accuracy** after preprocessing and PCA
- Regression models showed strong prediction performance (R² ≈ 0.92)
- CatBoost handled categorical features effectively
- Spectral Clustering performed better than basic clustering methods
- PCA significantly reduced training time while maintaining performance

-> Tools & Technologies

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib / Seaborn
- Google Colab
