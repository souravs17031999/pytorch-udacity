# PyTorch-Udacity
This is a repository for my progress on pytorch scholarship course on udacity.
The notebooks are not exactly same as Udacity notebook and i have created my own notebooks with modifications in each notebooks , some being slight like in lesson 2 and some being major like in lesson 4.
There are various introductory notebooks on topics like Tensors and working with tensors in Pytorch and various other topics will be added as i learn them.
link to udacity notebook - https://github.com/udacity/deep-learning-v2-pytorch
and link to course - https://in.udacity.com/course/deep-learning-pytorch--ud188
and other resources which are really helpful to learn are -
https://airtable.com/shrwVC7gPOuTJkxW0/tblUf4zxlIMLjwrbv

## Problems solved using neural networks:
### MNIST HANDWRITTEN DIGITS DATASET CLASSIFICATION PROBLEM - IN THIS WE ARE GIVEN 28\*28 GREY SCALE HANDWRITTEN IMAGES , 64 EXAMPLES AND WE ARE REQUIRED TO PREDICT THE DIGIT CORRECTLY BY FIRING AND ACTIVATING ONE OF THE NEURONS IN THE OUTPUT LAYER.
INPUT- 1 LAYER CONTAINING 784 NEURONS 
TWO HIDDEN LAYERS
ONE OUTPUT LAYER CONTAINING 10 NEURONS

### FASHION-MNIST DATASET CLASSIFICATION PROBLEM - IN THIS WE ARE GIVEN 28\*28 GERY SCALE FASHION CLOTHING IMAGES , 64 EXAMPLES AND WE ARE REQUIRED TO CORRECTLY CLASSIFY AS ONE OF THE 10 CLASSES (ANKIE BOOT , BAGS , SHIRTS , TROUSERS ETC...).
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

![neural networks](https://user-images.githubusercontent.com/33771969/50480811-382b0080-0a04-11e9-8e74-76f50ff9852a.jpeg)

Now let's break the mysterious black box of neural networks using example of MNIST dataset example:
* The idea is that we have 784 pixels of a image and each of 784 neurons in the input layer has some value (not necessarily same) which corresponds as 0 for black and 1 for white and others similarly from 0-1 depending on the contrast of pixels in the image.
Now the hidden layers neurons checks for some of the features of the image by breaking it down on the level of small edges which also have some pixel values stored in them and we apply a tranformation (WX+b) from input to hidden which corresponds to some combination of pixel values that we identify from image input.
Then in second hidden layer (if apply) we identify more of such features by applying another transformation and this goes on till the output layer which is the single neuron which makes final decision to predict the maximum probabilty of which of class it thinks our image belongs to.
Now we have our labels (we assign 10 labels from 0-9 to each of the class ) already , so we compare them with predicted values and calculates the error and further same backpropogation with stochastic gradient descent is applied and we find and settle at a minimum for cost function.
So , basically the transforms corresponds to some combination of pixels which in each layer fires the neurons in the next layer and goes on till output layer which finally combines the all features and predicts the class to which the image belongs may belong to.

Deep learning is a experimental field and creating a deep learning model or a neural network architechture is a art which is to be practised a lot.

### CAT_DOG CLASSIFIER WITH PRE-TRAINED MODEL - RESNET50
This model required 2048 neurons as input and we need just 2 neurons in the output layer to classify the image as cat or dog.
Then , we have 50 layers in total in this model and this model is taken torchvision package of pytorch.
The model is trained from data from ImageNet.
I made somechanges to classifier/fc (here) in this model and kept features parameters frozen (unchanged).
The benefit is that we need not train the whole neural network from our side and only classifier(fc) has to be trained and we just need to pass in our dataset on classifier(fc) and then it predicts with higher accuracies as is shown in the notebook.
For more info check out -
https://pytorch.org/docs/stable/torchvision/models.html

### CIFAR 10 CLASSIFIER 
USING TRANSFER LEARNING PRETRAINED MODEL RESNET152 AND CONVOLUTIONAL LAYERS IN USE.

### FLOWER IMAGE CLASSIFIER
USING TRANSFER LEARNING PRETRAINED MODEL RESNET152 AND CONVOLUTIONAL LAYERS IN USE.
