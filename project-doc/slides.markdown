% ID2223 Project
% Salman Niazi
% Jan 10,  2017

## Project Description
- Predict the solar radiation near Earth Surface   

## Data Samples
- 1 GB of CSV files with 0.4 million samples
- A typical sample looks like

|lev|p|T|q|lwhr|
|---|---|---|---|---|
|0|19.231|-80.0|0.0|0.122|
|1|57.692|-80.0|0.0|0.451|
|2|96.154|-70.874|0.029|-1.229|
|3|134.615|-51.083|0.262|-2.732|
|4|173.077|-36.489|0.977|-3.429|
|5|211.538|-25.816|2.211|-3.574|
|6|250.0|-17.87|3.756|-3.536|
|7|288.462|-10.404|5.431|-3.802|
|8|326.923|-6.608|4.226|-2.198|
|9|365.385|-2.388|8.776|-4.203|
|10|403.846|1.264|10.375|-3.567|
|11|442.308|4.462|11.895|-3.146|
|12|480.769|7.318|13.347|-2.829|
|13|519.231|9.903|14.733|-2.598|
|14|557.692|12.261|16.054|-2.397|
|15|596.154|14.421|15.778|-1.997|
|16|634.615|16.428|18.499|-2.239|
|17|673.077|18.304|19.648|-2.006|
|18|711.538|20.054|20.738|-1.908|
|19|750.0|20.443|20.147|-0.906|
|20|788.462|23.377|22.793|-1.692|
|21|826.923|24.749|23.775|-1.582|
|22|865.385|23.968|24.695|-0.116|
|23|903.846|27.491|25.599|-1.538|
|24|942.308|26.804|26.464|-0.053|
|25|980.769|29.988|27.285|-1.448|


## Propose Solution
- Regresstion Problem
	- with 26 outputs

- Could be implemented using 
	- Multivariate Regression
	- Feed Forward Neural Networks
	- **Convolution Neural Networks**

## Input
- Features
	- 26 values for T
	- 26 values for q
- Labels 
	- 26 lwhr values

## Input (Cont'd)
- The input can be morphed into 26 x 2 matrix 
	- did not produce very promissing results, as pooling can not shink the width of the input matrix. 
		- Min MSE achieved was 0.3

![26x2 Input Matrix](./img/26x2.png){height=200px}


## Input (Cont'd)
- The input can be morphed into 8 x 8 matrix 
	- padding is needed as there are only 52 input features 

![8 x 8 Input Matrix](./img/8x8.png){height=100px}

## Convolution Neural Network Model

![8 x 8 Input Matrix](./img/network.png)

## Model Complexity

|Layer|Size|Memory|Weights|Bias|
|-------|------------|-------------------------|-----------------------|----------------|
|Input|8x8x1|64|0|0|
|CONV|8x8x32|8x8x32 = 2048|2x2x1 x 32 = 128|32|
|POOL|8x8x32|8x8x32 = 2048|0|0|
|CONV|8x8x64|8x8x64 = 4096|2x2x1 * 64 = 256|64|
|POOL|4x4x64|4x4x64 = 512|0|0|
|CONV|4x4x128|4x4x128 = 2048|2x2x1 * 128 = 512|128|
|POOL|2x2x128|2x2x128 = 512|0|0|
|FC|1x512|512|2x2x128x512 = 262144|512|
|FC|1x256|256|512x256 = 131072|256|
|OUT|1x26|26|26x256 = 6656|26|

**Total memory  = 413908 x 4 bytes (*float32*) x 2 (back propagation) = 3311264 = 3.1 Megabytes**

## Evaluation Setup

- Training data set size 300,000 (75%). 
- Test data set size 100,000 (25%).
- Inputs are normalized using max-min scaling
	- X~norm~ = (X - X~min~) / ( X~max~ - X~min~)
- Learning Rate 0.05
- Dropout 0.95
- Max number of Epochs 30000
- Batch Size 10
- Weights were randomly initialized such that the random numbers had *mean=0.1* and *stddev=0.3*
- Bias were also randomly initialized such that the random numbers had *mean=0* and *stddev=0.03* 

## Results

![Mean Square Error of the CNN Model](./img/result1.png)

## Questions ?
