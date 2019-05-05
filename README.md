# Explore LSTM On Time Series Data (Without Any Code)
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![Generic badge](https://img.shields.io/badge/Status-Ready-Green.svg)](https://shields.io/)

---------------------------------------

![](http://www.microway.com/wp-content/uploads/TensorFlow_icon_180x148.png)

[![PyPI version](https://badge.fury.io/py/tensorflow-gpu%2F2.0.0a0.svg)](https://badge.fury.io/py/tensorflow-gpu%2F2.0.0a0)

# PoweredByTF 2.0 Challenge!
More news on [tensorflow.devpost.com](https://tensorflow.devpost.com/)


## Purpose

The sole purpose for the development of this software was to make it easy for learners to understand how tuning several hyperparameters can effect the result of an LSTM (Long Short Term Memory) network on various types of time series data.

## Dependencies

*   Python 3.x
*   Tensorflow
*   Matplotlib

## Running the App

*   Install `REQUIREMENTS.txt`, by running `pip install -r REQUIREMENTS.txt`
*   Open ``` app.py ``` file
*   Set the hyperparameter values ( **Dropout, Lag, Test Ratio, Max Epoch** )
*   Select a preloaded dataset 
*   Click ``` Start ```
*   To reset the console click ``` Reset ```

## OS Support

The application has been tested on Windows and Linux platforms. In case of any issue, feel free to raise an [issue](https://github.com/amitrajitbose/lstm-time-series-viz/issues/new).

## Examples
#### &nbsp;&nbsp;&nbsp;Increasing Sales dataset

&nbsp;&nbsp;&nbsp;&nbsp; <img src="https://raw.githubusercontent.com/amitrajitbose/lstm-time-series-viz/master/Observations/Increasing-Sales-Lag-10.png" alt="drawing" width="400"/>

#### &nbsp;&nbsp;&nbsp;Sinusoidal Curve Dataset

&nbsp;&nbsp;&nbsp;&nbsp; <img src="https://raw.githubusercontent.com/amitrajitbose/lstm-time-series-viz/master/Observations/Sine-Wave-Lag-2.png" alt="drawing" width="400"/>

## Model Architecture

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
unified_lstm (UnifiedLSTM)   (None, LAG-1, 30)             3840      
_________________________________________________________________
dropout (Dropout)            (None, LAG-1, 30)             0         
_________________________________________________________________
unified_lstm_1 (UnifiedLSTM) (None, 30)                    7320      
_________________________________________________________________
dense (Dense)                (None, 1)                     31        
=================================================================
Total params: 11,191
Trainable params: 11,191
Non-trainable params: 0
_________________________________________________________________
None

```

## Preloaded Datasets

Currently, the application supports 5 different datasets. We are going to add more datasets and probably improve the model in the next iteration of development. Contributions are welcomed.

- Sine Wave
- Cosine Wave
- Increasing Sales
- Decreasing Sales
- Random Data

## Developers
`Amitrajit Bose` + `Anirban Mukherjee`
