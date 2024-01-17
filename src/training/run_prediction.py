import torch
import lfcm
from collections import OrderedDict
import numpy as np
import customTransform
from dotenv import load_dotenv
import os

load_dotenv()

checkpoint_file = os.getenv('FCM_CP')

def image_to_tensor(filepath):
    image = np.zeros((3, 299, 299), dtype=np.float32)
    try:
        image = customTransform.filepath_to_image(filepath)
        image = customTransform.rescale(image, 299)
        image = customTransform.preprocess_image_to_np_arr(image)
    except:
        image = np.zeros((3, 299, 299), dtype=np.float32)

    return image

def text_to_tensor(id):
    text = np.zeros(150, dtype=np.float32)
    try:
        filepath = os.getenv('EMBED_TT_PATH')
        for i,line in enumerate(open(filepath)):
            data = line.strip().split(',')
            tweet_id = data[0]

            if id == tweet_id:
                arr = np.array(list(map(float, data[1:])))
                text = arr
    except Exception as e:
        text = np.zeros(150, dtype=np.float32)

    return text


def image_text_to_tensor(id):
    image_text = np.zeros(150, dtype=np.float32)
    try:
        filepath = os.getenv('EMBED_IT_PATH')
        for i,line in enumerate(open(filepath)):
            data = line.strip().split(',')
            tweet_id = data[0]

            if id == tweet_id:
                arr = np.array(list(map(float, data[1:])))
                image_text = arr
    except Exception as e:
        print('error:',e)
        image_text = np.zeros(150, dtype=np.float32)

    return image_text.copy()


def get_input_tensors(input_data):
    image = np.zeros((3, 299, 299), dtype=np.float32)
    text = np.zeros(150, dtype=np.float32)  # hidden state dimension of lstm
    image_text = np.zeros(150, dtype=np.float32)
    comments = np.zeros(150, dtype=np.float32)

    try:
        if "image_url" in input_data:
            image = image_to_tensor(input_data["image_url"])
        else:
            image = np.zeros((3, 299, 299), dtype=np.float32)
    except:
        image = np.zeros((3, 299, 299), dtype=np.float32)
    
    image = image.astype(np.float32)
    
    if 'tweet_id' in input_data:
        text = text_to_tensor(input_data['tweet_id'])
    else:
        text = torch.from_numpy(text.copy())
    
    text = text.astype(np.float32)

    if 'tweet_id' in input_data:
        image_text = image_text_to_tensor(input_data['tweet_id'])
    else:
        image_text = image_text.copy()

    image_text = image_text.astype(np.float32)

    comments = torch.from_numpy(comments.copy())
    
    image = torch.from_numpy(image.copy())
    text = torch.from_numpy(text.copy())
    image_text = torch.from_numpy(image_text.copy())

    return {
        "image": image,
        "image_text": image_text,
        "text": text,
        "comments": comments,
    }


def load_model():
    model = lfcm.OldModel(gpu=0)
    checkpoint = torch.load(checkpoint_file)
    model = lfcm.OldModel()
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda(0)
    model.load_state_dict(checkpoint)
    return model


def predict(model, input_tensors):
    in_ten = input_tensors
    image = in_ten["image"]
    image_text = in_ten["image_text"]
    text = in_ten["text"]

    with torch.no_grad():
        model.eval()
        output = model(image.unsqueeze(0), image_text.unsqueeze(0), text.unsqueeze(0))

    return output

model = load_model()