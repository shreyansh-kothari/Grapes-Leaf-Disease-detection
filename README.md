# Grapes-Leaf-Disease-detection
Built a deep learning model using tensorflow and keras for grape leaf disease detection

# Model Accuracies
1. Coloured ~ 96-97%
2. Grey ~ 78-80%

# Link to the dataset (Images and pickle files)
https://drive.google.com/open?id=1m7umLLc8sXCwcsrQF43FOxirHZiT8ev6

# The codes have comments. Please read the comments for understading the code

# Description of the files
1. Coloured Model Folder - containes a saved trained model and the raw jupyter notebook for coloured model
2. Grey Model Folder - containes a saved trained model and the raw jupyter notebook for grey model
3. CNN-leaf disease detection coloured.py - python file of coloured model with comments and explanation of code
4. CNN-leaf disease detection grey.py - python file with of grey model comments and explanation of code

# Executing the code
1. Install the libraries used in the code
2. Change the paths in the code as suggested in the comments
3. Follow the comments in the code to get better understanding and running the code

# Concepts  used (basics)
1. tensorflow 2.0 (gpu version)
2. Numpy
3. Pandas
4. Matplotlib
5. cv2 

# Explanation
1. First the images from the dataset are read and converted to arrays and stored with label
2. Resizing the images 
3. Creating the pickle files for future use
4. Reshaping the data for model compatability
5. Converting the labels to categories for model compatability
6. Constructing the neural network using convolution, maxpooling and dense layers
7. Fitting the model with 15% shuffling validation data
8. Saving the model
9. Evaluating the model on test data
10. Predicting a sample image


