# PyTorch-Udacity
This is a repository for my progress on pytorch scholarship course on udacity.
The notebooks are not exactly same as Udacity notebook and i have created my own notebooks with modifications in each notebooks , some being slight like in lesson 2 and some being major like in lesson 4.
There are various introductory notebooks on topics like Tensors and working with tensors in Pytorch and various other topics will be added as i learn them.
link to udacity notebook - https://github.com/udacity/deep-learning-v2-pytorch
and link to course - https://in.udacity.com/course/deep-learning-pytorch--ud188
and other resources which are really helpful to learn are -
https://airtable.com/shrwVC7gPOuTJkxW0/tblUf4zxlIMLjwrbv

Two problems solved till now :
* MNIST HANDWRITTEN DIGITS DATASET CLASSIFICATION PROBLEM - IN THIS WE ARE GIVEN 28\*28 GREY SCALE HANDWRITTEN IMAGES , 64 EXAMPLES AND WE ARE REQUIRED TO PREDICT THE DIGIT CORRECTLY BY FIRING AND ACTIVATING ONE OF THE NEURONS IN THE OUTPUT LAYER.
INPUT- 1 LAYER CONTAINING 784 NEURONS 
TWO HIDDEN LAYERS
ONE OUTPUT LAYER CONTAINING 10 NEURONS

* FASHION-MNIST DATASET CLASSIFICATION PROBLEM - IN THIS WE ARE GIVEN 28\*28 GERY SCALE FASHION CLOTHING IMAGES , 64 EXAMPLES AND WE ARE REQUIRED TO CORRECTLY CLASSIFY AS ONE OF THE 10 CLASSES (ANKIE BOOT , BAGS , SHIRTS , TROUSERS ETC...).
INPUT- 1 LAYER CONTAINING 784 NEURONS
1 HIDDEN LAYER CONTAINING 256 NEURONS
1 OUTPUT LAYER CONTAINING 10 NEURONS.

[In both of the above problems we take 28\*28 grey scale images so that we take each pixel values.Now the input tensor from MNIST dataset is of shape\[64,1,28,28\] where 64 is no of examples in a batch, 1 is color channel , 28 is height and 28 is width.So we change and flatten this into shape\[64,784\].The idea is to pick all the combinations of pixel values each time passing them to 784 neurons in the input layer  and train them to predict correct scores as compared to actual labels as much accurate as it could be.
The abstracted idea is to define a line or a boundary (non-linear) which separates a given set of classes based on some features.
The brief idea - 
We pass the input to first layers and then do some linear transformation (xW+b)and then pass them to some activation function (here , ReLU), then pass them into another hidden layers and then repeat it until we finally decide the class 0-9 and activate the particular neuron the model thinks it to be in the output layer and passing through softmax for a probability distribution for multiclassification problem.This is generally called forward pass in neural networking.
Then have a loss function (error function) which calculates how far our predictions are from actual labels (here cross entropy loss).
Then we gotta calculate gradient of all the steps from backwards in order to minimize this loss function and finally move into the direction where this gradient is decreasing to get a minimum.
In this way , we have modified our original parameters (weights) and updated them so that they predict the scores more accurately next time.
We do this number of items over several batches known as epochs.
Now this is all called backpropogation in neural networking.
After training the model over the training dataset , we set out to test our model against the a new dataset called validation dataset for comparing it with some new data.
Then , we can finally use our model for inherence that is for making predictions over real-world dataset.

"The lesson 4 module contains videos of neural networks into play for both of the above problems solved" 

https://user-images.githubusercontent.com/33771969/50480811-382b0080-0a04-11e9-8e74-76f50ff9852a.jpeg

