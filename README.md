# **ID2223 Project**
# Group members: Salman Niazi and Shadi Issa


## Project Description
The goal of the project is to predict the solar radiation level near the Earth’s surface. The dataset consists of 0.4 million labeled data samples stored in individual files in CSV format. A data sample looks like the following.

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

The first two columns (**lev**, **p**) are static, that is, the values of these columns in all the data samples are the same. Therefore, the first two columns does not add contain any useful information that can be used to predict the final labels. The column **T** represents the temperature, **q** represents the pressure and the column **lwhr** represents the radiation. The **lwhr** is the label column. Each CSV file contains exactly 26 data points representing temperature, pressure and solar radiation at the Earth's surface at 26 different heights. For example, in the above data the 26th data point shows that the temperature near the Earth surface is 29.988, the pressure is 27.285 and the solar radiation is -1.448.

### Feature Normalization

The input features and the label values vary quite a lot that causes the gradient to fluctuate. All the input features and labels are normalized using min-max scaling. Which is defined as

X~norm~ = (X - X~min~) / ( X~max~ - X~min~)

|Feature|Min|Max|
|--------|-------|------|
|**T**|-80.0|29.988|
|**q**|0.0|27.285|
|**lwhr**|-9.94|6.69|


## Proposed Solutions
The goal of the project is to predict the solar radiation near the Earth's surface using the temperature and pressure values. This is a *regression* problem with 26 output labels. The problem can be solved using machine learning techniques, such as, *multivariate regression* and deep learning techniques, such as, *feed forward neural networks* and *convolution neural networks*.

We have implemented the solution using the *feed forward neural networks* and *convolution neural networks*.


## Initial Solution
According to the universal approximation theorem, Feedforward neural networks with a single
hidden layer can approximate continuous functions. In our problem we are trying to replace an
analytical model to enhance the performance; hence a feedforward
neural network with a single hidden layer is ought to be a suitable solution.

### Feedforward Neural Network Model

![Architecture of feedforward neural network \label{fig:ff}](./img/ff.png)

Figure \ref{fig:ff} shows the model of our feedforward network. It is composed of a single
hidden layer with 20 neurons and an output layer of a single linear neuron. Neurons within the
hidden layer use a sigmoid activation function. The weights of the hidden neurons has the
structure $26\times1$, while the weights of the output neuron has the structure $1\times26$
to produce the output \emph{lwhr}. Both hidden and output layer have a bias that is initialized to zero.

### Evaluation

The model was implemented using Tensorflow running in a VMware virtual machine which is install on a 2 core
Macbook. The following are the values for different parameters obtained after optimization.

- Training data set size 1,400,000 (70%).
- Test data set size 600,000 (30%).
- Max number of Epochs 14000
- Batch Size 100
- Weights and biases are initialized to zeros

![Mean Square Error of the FFN Model \label{fig:result_ff}](./img/result_ff.png)

![Sample prediction of FFN \label{fig:result_ff_1}](./img/result_ff_1.png)

![Sample prediction of FFN \label{fig:result_ff_2}](./img/result_ff_2.png)

Figure \ref{fig:result_ff} shows the Mean Square Error *MSE* against the number of epochs. We can see that the MSE
decreases exponentially as more training batches are performed until it stabilizes at a MSE of 0.000491042. However,
despite the very low MSE, when we look at some sample results such as in Figure \ref{fig:result_ff_1} and Figure
\ref{fig:result_ff_2} we can see that our model is not able to predict the local spikes. To overcome this limitation,
we opted to go for another solution based on convolution neural networks.



## Convolution Neural Network Solution

Next, we choose convolution neural networks *(CNN)* as CNN are suitable for problems when there is a correlation between the input features. In our case the input feature are corelated by height, that is the samples are collection at 26 different distances from the earth surface and the measurements of the features and the labels gradually change.  

### Input
The convolutin neural network expects an input matrix of size *m x n* size. We could combine the **T** and **q** columns to form a 26x2 matrix as shown below.

![26x2 Input Matrix \label{fig:input}](./img/26x2.png){height=200px}

This 26x2 input matrix did not produce very promissing results, as pooling can not shink the width of the input matrix. The minimum MSE achieved uisng this input format was 0.3.

The input can be morphed into 8 x 8 matrix with zero padding as there are only 52 input features

![8 x 8 Input Matrix \label{fig:input_mat}](./img/8x8.png){height=100px}

### Convolution Neural Network Model

![Convolution Neural Network Model \label{fig:cnn}](./img/network.png)

The model of the convolution neural network model is shown in Figure \ref{fig:cnn}. It consists of three convolution layers, two fully connected layers and an output layer.

- **Layer 1 (Convolution):** The first layer is a convolution layer that uses *2x2* filter with *32* features. The layer is followed by a ReLU normalization layer and max pooling layer with a stride of *1*.
- **Layer 2 (Convolution):** The second layer is a convolution layer that uses *2x2* filter with *64* features. The layer is followed by a ReLU normalization layer and max pooling layer with a stride of *2*.
- **Layer 3 (Convolution):** The third layer is a convolution layer that uses *2x2* filter with *128* features. The layer is followed by a ReLU normalization layer and max pooling layer with a stride of *2*.
- **Layer 4 (Fully Connected Layer):** The fourth layer is fully connected layer with 512 neurons using ReLU activation function.
- **Layer 5 (Fully Connected Layer):** The fourth layer is fully connected layer with 256 neurons using ReLU activation function.
- **Layer 6 (Output Layer):** The last layer is 26 neuron output layer.


### Regularization

For regularization *dropout* is used in the last fully connected layer.  The dropout value was set to *0.95*, that is, during each training epoch 5% of the neurons in the last fully connected layer are randomly set inactive. This enables all neurons to equally learn the model.

### Model Complexity

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

## Leaky ReLU

During our evaluation we found that the CNN also could not predict the spikes. The main cause of the problem was dead neurons in the model.

"ReLU units can be fragile during training and can “die”. For example, a large gradient flowing through a ReLU neuron could cause the weights to update in such a way that the neuron will never activate on any datapoint again." [^dr]

Dead neurons problem can be solved using Leaky ReLU abstraction. In Leaky ReLU the sloap was set ot 0.001.


[^dr]: http://cs231n.github.io/neural-networks-1/

## Evaluation

The model was implemented using Tensorflow running in a docker instance. The docker instance was run on HP ProLiant DL360p Gen8 with 32 cores and 256 GB of RAM. Following are the values for different parameters values obtained after hyper-parameter optimization.

- Training data set size 360,000.
- Test data set size 40,000.
- Learning Rate 0.001
- Dropout 0.95
- Max number of Epochs 120000
- Batch Size 3
- Weights were randomly initialized such that the random numbers had *mean=0.1* and *stddev=0.3*
- Bias were also randomly initialized such that the random numbers had *mean=0* and *stddev=0.03*


Following figures show a sample predictions of our model. With all the optimization the model accurately predicts the output including the spikes in the labels. The square error of each prediction is written under the graphs.


![Squared Error: 0.00235824](./img/p1.png)


![Squared Error: 0.00270954](./img/p2.png)


![Squared Error: 0.00112128](./img/p3.png)


![Squared Error: 0.0198945](./img/p4.png)
