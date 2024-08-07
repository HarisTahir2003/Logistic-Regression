# Hi, I'm Haris! 👋


[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/) 


# Logistic Regression Assignment

This Jupyter Notebook explores the concepts of Logistic Regression and Ridge Regression, with a particular focus on building a classifier to categorize celestial objects as either stars, galaxies, or quasars. The notebook is structured to provide a comprehensive understanding of these regression techniques, and includes practical implementations, visualizations, and model evaluations. <br> 

The Logistic Regression folder contains the following files:
- A .ipynb file (Jupyter Notebook) that contains all the code regarding the assignment including text blocks explaining portions of the code
- A corresponding .py file
- seven .png files that are screenshots of the plots in the Jupyter Notebook
- a star_classification.csv file that contains the raw data for the Logistic Regression part of the assignment
- an auto_mpg.csv file that contains the raw data for the Ridge Regression part of the assignmenton. You can download and save the .csv files in your computer and open them as Excel files for better readability.


## Table of Contents

1. [Introduction](#introduction)
2. [Installation Requirements](#installation-requirements)
3. [Project Structure](#project-structure)
4. [Data](#data)
5. [Training and Evaluation](#training-and-visualization)
6. [Lessons](#lessons)
7. [Screenshots](#screenshots)
   
## Introduction

Logistic regression is a Machine Learning technique used for binary or multi-class classification that models the relationship between a dependent binary/multi-class variable and one or more independent variables. Unlike linear regression, which predicts continuous outcomes, logistic regression predicts the probability of an outcome.

 This assignment provides a clear and concise example of how to implement multi-class logistic regression from scratch using Python.
## Installation Requirements

To run this notebook, you will need the following packages:
- numpy
- pandas
- matplotlib
- scikit-learn

You can install these packages using pip:

```bash
 pip install numpy
```
```bash
 pip install pandas
```
```bash
 pip install matplotlib 
```
```bash
 pip install scikit-learn
```

Useful Links for installing Jupyter Notebook:
- https://youtube.com/watch?v=K0B2P1Zpdqs  (MacOS)
- https://www.youtube.com/watch?v=9V7AoX0TvSM (Windows)

It's recommended to run this notebook in a conda environment to avoid dependency conflicts and to ensure smooth execution.
<h4> Conda Environment Setup </h4>
<ul> 
   <li> Install conda </li>
   <li> Open a terminal/command prompt window in the assignment folder. </li>
   <li> Run the following command to create an isolated conda environment titled AI_env with the required packages installed: conda env create -f environment.yml </li>
   <li> Open or restart your Jupyter Notebook server or VSCode to select this environment as the kernel for your notebook. </li>
   <li> Verify the installation by running: conda list -n AI_env </li>
   <li> Install conda </li>
</ul>


## Project Structure

The notebook is organized into the following sections:
<ul>
<li> Introduction: Overview of the project and logistic regression.  </li> <br> 
   
<li> Task 1A: Multinomial Logistic Regression <br>
&emsp; 1) Data Loading and Preprocessing: Steps to load and preprocess the dataset. <br>
&emsp; 2) Model Training: Training the logistic regression model. <br>
&emsp; 3) Model Evaluation: Evaluating the model performance. </li> <br> 


<li> Task 1B: Reflection Questions related to the logistic regression task </li> <br> 
<li>
Task 2: Ridge Regression <br>
&emsp; 1) Data Loading and Preprocessing: Steps to load and preprocess the dataset. <br>
&emsp; 2) Model Training: Training the ridge regression model. <br>
&emsp; 3) Model Evaluation: Evaluating the model performance using plots and some reflection questions </li> <br>

</ul>

## Data

The dataset for the first task is provided in a csv file titled `star_classification.csv`. It consists of 100,000 observations of space taken by the SDSS (Sloan Digital Sky Survey). Every observation is described by 17 feature columns and 1 class column which identifies it to be either a star, galaxy or quasar.  <br> 

 Dataset Summary:
- **Feature Type:** Varies
- **Instances:** 100,000
- **Input Features:** 17 
- **Output:** Class
<br>
The dataset for the second part is provided in a csv file titled `auto_mpg.csv`. <br>
 Dataset Summary:
<ul>
<li> Feature Type: Varies </li>
<li> Instances: 398 </li>
<li> Input Feature: mpg </li>
<li> Output: displacement </li>
</ul>


## Training and Visualization

The entire training process alongside the maths involved is explained in detail in the jupyter notebook. 
- Note: You need to be proficient in Calculus to fully understand the gradient descent algorithm, especially the concept of partial derivatives. Additionally, a good knowledge of Linear Algebra is required to understand the various matrix and vector operations that are performed in the assignment.


## Lessons

A logistic regression project can teach a variety of valuable skills and concepts, including:

- Data Preprocessing: How to clean and prepare data for analysis, including handling missing values, scaling features, and encoding categorical variables.

- Feature Selection: Identifying which features (variables) are most important for making predictions and how to choose them effectively.

- Model Building: Understanding how to build a logistic regression model, including splitting data into training and testing sets, fitting the model, and predicting outcomes.

- Performance Evaluation: Using metrics like Root Mean Squared Error (RMSE) to evaluate the performance of your model and understand its accuracy.

- Interpreting Results: Understanding the results of the logistic regression model and what they signify.

- Algorithm Implementation: Learning about the underlying algorithm used in linear regression and how it optimizes the line of best fit.


## Screenshots
<h3> Ridge Regression </h3>
<h4> 1. This image shows how the value of the Root-Mean-Square-Error changes for various training and testing datasets as the value of the regularization parameter (lambda) is gradually increased from 0 to 10. The four datasets include the training and testing datasets of each of the analytical and gradient-descent solutions. </h4>
<img src="pic1.png" width="450px"> <br> 

<h4> 2. This image shows the output of the regression model with the least validation Root-Mean-Square-Error overlaid on top of the original mpg vs displacement data. </h4>
<img src="pic2.png" width="450px"> <br> 





## License

[MIT](https://choosealicense.com/licenses/mit/)
