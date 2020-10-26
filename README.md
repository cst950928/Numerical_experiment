

## Introduction
This folder contains the datasets and source code (i.e. .py files) used in the paper “On the Cluster-aware Supervised Learning (CluSL): Frameworks, Convergent Algorithms, and Applications.” The paper proposed two frameworks (algorithms), corresponding to subfolders Experiment 1-6 and subfolders Experiment 7-8, respectively. To successfully run the code in subfolders, you may need to install Python (i.e. .py files) and Matlab (i.e. .m files). The paths used to process the data in the codes should be revised accordingly.

Below is the detailed explanation of each subfolder.

## Subfolder: Experiment 1
 - **Purpose:** Experiment 1 was implemented to test the performance of the first algorithm in the paper and compare it with non-convex solver Knitro implemented in Matlab.
- **Data Set Info:** The data file contains nine sets of hypothetical data generated using the given distribution displayed in the paper. Sheet1-Sheet3, Sheet4-Sheet6, Sheet7-Sheet9 include three sets of one-dimensional, two-dimensional, and three-dimensional data points correspondingly, where the last column of each sheet is regarded as the response variable.
- **Source code Info:** The code folder includes one .py file and one .m file. "Experiment 1. py" is the code of the first algorithm in the paper. "CluSL.m" is the Matlab code of solver Knitro.

## Subfolder: Experiment 2
- **Purpose:** Experiment 2 was implemented to test the performance of the first framework for the regression problem in the paper using Hypothetical data.
- **Data Set Info:** Hypothetical data was generated using a piece-wise function explained in the paper. The data file includes 500 training data points and 9 testing datasets (500 data points each).

## Subfolder: Experiment 3
- **Purpose:** Experiment 3 was implemented to test the first framework’s performance for the regression problem using UCI data.
- **Data Set Info:** Three files contain datasets used in real estate valuation, red wine, and white wine quality cases. The source data are available at https://archive.ics.uci.edu/ml/index.php.
- **Source code Info:** The code folder includes three py files. "Experiment 3_RAM.py" is the code of the proposed algorithm in the paper. "Experiment 3_RF.py" and "Experiment 3_SVR.py" are open-sourced Random Forest and Support Vector Machine codes in the scikit-learn package.

## Subfolder: Experiment 4
- **Purpose:** Experiment 4 was implemented to demonstrate the interpretable results yielded by the first framework in the paper using a practical regression problem.
- **Data Set Info:** The "housing.csv" used in Experiment 4 contains 20640 data points with 9 features. Data cleaning should be implemented according to the paper. More detailed information is available at [https://www.kaggle.com/camnugent/California-housing-prices](https://www.kaggle.com/camnugent/California-housing-prices)

## Subfolder: Experiment 5
- **Purpose:** Experiment 5 was implemented to test the performance of the first framework in the classification problem based on low-dimensional UCI datasets.
- **Data Set Info:** Five files contain datasets used in "Contraceptive Method Choice", "Steel plates faults", "Wireless indoor localization", "White wine quality 5 classes", and "Yeast" cases. The source data are available at https://archive.ics.uci.edu/ml/index.php. Please note that in the white wine case, we removed level 3 and 9 which contain very few data.
- **Source code Info:** The code folder includes two .py files. "Experiment 5_Clu-MSVM.py" is the code of the proposed algorithm in the paper. "Experiment 5_SVC.py" is an open-sourced Support Vector Classification code in the scikit-learn package.

## Subfolder: Experiment 6
- **Purpose:** Experiment 6 was implemented to test the performance of the first framework in the classification problem based on high-dimensional image datasets.
- **Data Set Info:** Three files contain three datasets used in "CIFAR-10", "Magnetic-tile-defect", and "MNIST handwritten digits" cases. Please note that we preprocessed the second dataset images by uniformly resizing all the images into 50×50 pixels. The source data are available at [http://www.cs.toronto.edu/~kriz/cifar.html](http://www.cs.toronto.edu/~kriz/cifar.html)
[https://github.com/abin24/Magnetic-tile-defect-datasets.](https://github.com/abin24/Magnetic-tile-defect-datasets.)  and http://yann.lecun.com/exdb/mnist/.
- **Source code Info:** The code folder includes three .py files. "Experiment 6_Clu-CNN.py" is the code of the first algorithm in the paper. "Experiment 6_CNN.py" is an open-sourced Convolutional Neural Networks code in the Python deep learning API: Keras. "read_data.py" includes some preprocessing process of image data.

## Subfolder: Experiment 7
- **Purpose:** Experiment 7 was implemented to show both the classification performance and interpretability of the second framework based on small-sized image datasets.
- **Data Set Info:** "digital.xlsx" contains both the training and testing images in the "Semeion Handwritten Digit" case. The size of each image is 16×16 grayscale values. "training.xlsx" and "testing.xlsx" contains the training and testing images in the "Optical Recognition of Handwritten Digits" case. The size of each image is 8×8. The source data are available at https://archive.ics.uci.edu/ml/index.php.
- **Source code Info:** The code folder includes two .py files. "Experiment 7_F-RAM.py" is the code of the second framework and algorithm in the paper. "Experiment 7_CNN.py" is open-sourced Convolutional Neural Networks code in the Python deep learning API: Keras.

## Subfolder: Experiment 8
- **Purpose:** Experiment 8 was implemented to show both the classification performance of the second framework based on larger-sized image datasets.
- **Data Set Info:**  Please refer to Data Set Info in folder Experiment 6.
- **Source code Info:** The code folder includes three .py files. "Experiment 8_FCluSL.py" is the code of the second framework and algorithm in the paper. "Experiment 8_CNN.py" is an open-sourced Convolutional Neural Networks code in the Python deep learning API: Keras. "read_data.py" includes some preprocessing process of image data.


Feel free to refer to the data and codes displayed in the subfolders. If used in your research, please cite our paper. 
## References:
Chen, S., Xie, W. (2020). On the Cluster-aware Supervised Learning (CluSL): Frameworks, Convergent Algorithms, and Applications. *INFORMS Journal on Computing*. Accepted.
