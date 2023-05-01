DEVELOP_MODE = False 
USER_MODE = not DEVELOP_MODE
AZURE_SEARCH_KEY = ""

import os
from pathlib import Path
import gradio as gr
from fastai.vision.all import *


if DEVELOP_MODE:
    import fastbook
    from fastbook import *
    from fastai.vision.widgets import *
    from fastai.vision.all import *
    fastbook.setup_book()

    import uuid
    import requests
    import imghdr
    from PIL import Image
    import numpy as np
    

attn_slicing_enabled = True


def download_unique_image(url, folder_path):
    try:
        response = requests.get(url, timeout=10)
        content_type = response.headers.get('Content-Type')
        if content_type.startswith('image'):
            image_type = imghdr.what(None, response.content)
            if image_type == 'jpeg':
                extension = 'jpg'
            else:
                extension = image_type
            filename = str(uuid.uuid4()) + '.' + extension
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'wb') as f:
                f.write(response.content)
    except:
        pass


def remove_corrupted_images(folder_path):
    count = 0
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            with Image.open(file_path) as img:
                pass
        except Exception as err:
            os.remove(file_path)
            count += 1


def normalize_dog_name(dog_name):
    return dog_name.replace(' ', '_').lower()


def download_images_():
    dogs = {
        'Zwergspitz Dog': [],
        'Bouledogue Français Dog': [],
        'Shih Tzu Dog': [],
        'Rottweiler Dog': [],
        'Pug Dog': [],
        'Golden Retriever Dog': [],
        'Deutscher Schäferhund Dog': [],
        'Yorkshire Terrier Dog': [],
        'Border Collie Dog': [],
        'Dachshund Dog': [],
        'Poodle Dog': [],
        'Labrador Retriever Dog': [],
        'Pinscher Dog': [],
        'Golden Retriever': [],
    }
    DOGS_NAMES = tuple(dogs.keys())
    if DEVELOP_MODE:
        if not PATH.exists():
            PATH.mkdir()
            for dog_name in DOGS_NAMES:
                urls = search_images_bing(
                    AZURE_KEY, dog_name).attrgot('contentUrl')
                dogs[dog_name] = urls

                dest = os.path.join(PATH, normalize_dog_name(dog_name))
                if not os.path.exists(dest):
                    os.mkdir(dest)
                download_images(dest, urls=urls)
                remove_corrupted_images(dest)
    return [dog.replace('Dog', '') for dog in DOGS_NAMES]


def train_model():
    dogs_datablock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(128, ResizeMethod.Squish),
                   Resize(128, ResizeMethod.Pad, pad_mode='zeros'),
                   RandomResizedCrop(128, min_scale=0.3),
                   ]
    )
    dogs_dataloaders = dogs_datablock.dataloaders(PATH)
    # dogs_dataloaders = dogs_dataloaders.new(
    #     item_tfms=Resize(128, ResizeMethod.Squish))
    learn_ = vision_learner(dogs_dataloaders, resnet18, metrics=error_rate)
    learn_.fine_tune(4)
    learn_.export('dogs.pkl')
    return learn_


def classify_image(image):
    global learing
    pred, pred_idx, probs = learing.predict(image)
    return f"Prediction: {pred.replace('_', '').replace('dog', '').title()};\n Probability: {probs[pred_idx]:.04f}"


def get_model_():
    path = Path()
    model = None

    if any(file.endswith('.pkl') for file in os.listdir(path)):
        model_ = load_learner('dogs.pkl')
    else:
        model_ = train_model()
    return model_


AZURE_KEY = os.environ.get(
    'AZURE_SEARCH_KEY',
    AZURE_SEARCH_KEY,
)
PATH = Path('dogs')

dogs = download_images_()
learing = get_model_()


# Gradio
iface = gr.Interface(
    classify_image,
    inputs="image",
    outputs="text",
    title="Classificação de Imagens",
    description="Insira uma imagem para ser classificada"
)


def set_mem_optimizations(pipe):
    if attn_slicing_enabled:
        pipe.enable_attention_slicing()
    else:
        pipe.disable_attention_slicing()


def list_breeds():
    global dogs
    html = "<div class='row'>"
    html += "<div class='column'>"
    html += "<h2>List of breed dogs trained:</h2>"
    html += "<ol>" + "".join([f"<li>{breed}</li>" for breed in dogs]) + "</ol>"
    html += "</div>"
    html += "<div class='column'>"
    html += "<h2>Author:</h2>"
    html += "<a href='https://github.com/ericxlima'><img src='https://avatars.githubusercontent.com/u/58092119?v=4' alt='profile image' style='width:40%' /></a>"
    html += "<h2><a href='https://github.com/ericxlima'>Eric de Lima</a></h2>"
    html += "</div>"
    html += "</div>"
    return html 


image = gr.Image(shape=(224, 224))
label = gr.Label(num_top_classes=3)
breeds_list = list_breeds()

demo = gr.Interface(
    fn=classify_image,
    inputs=image,
    outputs=label,
    title="🐶 Dog Breed Classifier",
    interpretation="default",
    description="Upload an image of a dog and the model will predict its breed.",
    article=breeds_list,
    css=".row { display: flex; } .column { flex: 50%; }",
)

demo.launch(share=True, debug=True)
