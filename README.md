## Dog Breed Classifier
#

<a target="_blank" href="https://colab.research.google.com/github/ericxlima/dog-breed-classifier/blob/main/Copy_of_Projetin_Dog_Breed_Classifier_Deploy.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

This project is a GUI (Graphical User Interface) for classifying dog breed by images. It uses a pre-trained neural network called ResNet-18 to perform the classification.

In the develop mode, the code has 3 main functions:

    1. `download_images()`: Download images of dogs from the internet and store them locally, using AZURE BING SEARCH API.

    2. `train_model()`: Trains the dog image classification model using transfer learning on top of the pre-trained ResNet-18.

    3. `classify_image()`: Uses the trained model to classify a dog image.

In the default mode, the code has only `classify_image()`, and the model is downloaled automatic.

The code also has a Gradio interface that allows the user to load a dog image and get the model's classification. The model classifies images into 13 different dog breeds:

![img](https://imgur.com/P3HGygv.png)

The interface also has a memory optimization option that can be enabled or disabled by the user. When the option is enabled, the template uses less memory but takes a little longer to sort the image. The option is disabled by default.
User

![img](https://imgur.com/tjtQfER.png)
