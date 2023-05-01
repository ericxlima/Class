## Dog Breed Classifier
#

This project is a GUI (Graphical User Interface) for classifying dog breed by images. It uses a pre-trained neural network called ResNet-18 to perform the classification.

In the develop mode, the code has 3 main functions:

    1. `download_images()`: Download images of dogs from the internet and store them locally, using AZURE BING SEARCH API.

    2. `train_model()`: Trains the dog image classification model using transfer learning on top of the pre-trained ResNet-18.

    3. `classify_image()`: Uses the trained model to classify a dog image.

In the default mode, the code has only `classify_image()`, and the model is downloaled automatic.

The code also has a Gradio interface that allows the user to load a dog image and get the model's classification. The model classifies images into 13 different dog breeds:

- Zwergspitz
- French Bulldog
- Shih Tzu
- Rottweiler
- Pug
- Golden retriever
- Deutscher Schäferhund
- yorkshire terrier
- border collie
- Dachshund
- Poodle
- Labrador Retriever
- Pinscher

The interface also has a memory optimization option that can be enabled or disabled by the user. When the option is enabled, the template uses less memory but takes a little longer to sort the image. The option is disabled by default.
User

