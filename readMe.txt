

1. Load 4 programs into Matlab.(loadMNISTImages.m, loadMNISTLabels.m, Project1_Steepest.m, Project1_Optimization.m)

2. Test on the Iris dataset. 
	1) Load Iris dataset from file. 
	2) In the command line window of Matlab, input "w = Project1_Steepest(iris_data,regularize)" to test.
	   The "regularize" can be either true or false to decide whether the model uses regularzer or not.
	3) In the command line window of Matlab, input "w = Project1_Optimization(iris_data,regularize)" to test.
	   The "regularize" can be either true or false to decide whether the model uses regularzer or not.

3. Test on the MNIST dataset.
	1) Call function from "loadMNISTImages.m" to load all images from file.
	2) Call function from "loadMNISTLabels.m" to load all labels from file.
	3) Concatenate all data with all labels.(mnist_data = cat(2,data,labels))
	4) In the command line window of Matlab, input "w = Project1_Steepest(mnist_data,regularize)" to test.
	   The "regularize" can be either true or false to decide whether the model uses regularzer or not.
	5) In the command line window of Matlab, input "w = Project1_Optimization(mnist_data,regularize)" to test.
	   The "regularize" can be either true or false to decide whether the model uses regularzer or not.

4. Test on the Cifar-10 dataset.
	1) Load all data from a Cifra-10 data batch file. 
	2) Use PCA to get all coefficients. (coeff = pca(cifar_data);)
	3) Specify the number of reduced dimensions. (reducedDim = coeff(:,1:numberDim);)
	4) Generate all data with reduced dimensions. (reducedData = cifar_data * reducedDim;)
	4) In the command line window of Matlab, input "w = Project1_Steepest(reducedDdata,regularize)" to test.
	   The "regularize" can be either true or false to decide whether the model uses regularzer or not.
	5) In the command line window of Matlab, input "w = Project1_Optimization(reducedData,regularize)" to test.
	   The "regularize" can be either true or false to decide whether the model uses regularzer or not.
