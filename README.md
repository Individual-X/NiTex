# Pupose

Assesment for Nitex ML engineer.

## Installation

1. Clone the repository.
2. Install required dependencies from `requirements.txt`.
3. Run the evaluate_model.py to see the accuracy score on test data set.

## DataSet insight

By analyzing the dataset I came across that the this is a Synthetic data set having equal distribution of all classes. 

## Model  Selection

For this kind of casses when it comes to handling the image datasets SOA(Stat of Art) algorithms like CNN (Convolutional Neural Networks) perform best.
As this architecture stacked with:
1. Convolutional Layer
2. Pooling Layer
3. Fully Connected Layer
4. Dropout
5. Activation Functions
   

## Model Optimization

### Reguralization
I used Reguralization to calibrate model in order to minimize the adjusted loss function and prevent overfitting or underfitting.

### Early stooping

In this work to choose optimal number of epoch for saving computational time, resources and over or underfitting I used val_accuracy to observe the network learning pattern and stop when the model heading to take weird decision.

## Argument Instruction

python evaluate_model.py --test-data "test.csv" if you want different test different test dataset.
