import onnxruntime as rt
import numpy as np
import pandas as pd

import os
import glob
import io

from PIL import Image
from IPython.display import HTML
from base64 import b64encode
from sklearn.metrics.pairwise import cosine_similarity
from icecream import ic

from Utils import dbimutils

def get_labels(csv_path):
    csv_path = csv_path
    dataframe = pd.read_csv(csv_path)

    tag_list = dataframe["name"].tolist()
    rating_indices = list(np.where(dataframe["category"] == 9)[0])
    general_indices = list(np.where(dataframe["category"] == 0)[0])
    character_indices = list(np.where(dataframe["category"] == 4)[0])

    return tag_list, rating_indices, general_indices, character_indices

def get_scores(probs, general_threshold, character_threshold, tag_names: list[str], rating_indexes: list[np.int64], general_indexes: list[np.int64], character_indexes: list[np.int64]):
    labels = list(zip(tag_names, probs[0].astype(float)))

    # rating dict
    rating_names = [labels[i] for i in rating_indexes]
    rating = dict(rating_names)

    # general dict
    general_names = [labels[i] for i in general_indexes]
    general_result = [x for x in general_names if x[1] > general_threshold]
    general_result = dict(general_result)

    # characters_dict
    character_names = [labels[i] for i in character_indexes]
    character_result = [x for x in character_names if x[1] > character_threshold]
    character_result = dict(character_result)

    b = dict(sorted(general_result.items(), key=lambda item: item[1], reverse=True))
    tag_strings = (", ".join(list(b.keys())).replace("_", " ").replace("(", "\(").replace(")", "\)"))
    raw_tag_strings = ", ".join(list(b.keys()))

    return tag_strings, raw_tag_strings, rating, general_result, character_result

def get_img_list(dir):
    img_path_list = []
    patterns = (".png", ".jpg", ".jpeg", ".PNG", ".JPEG", ".JPG")

    for file in glob.glob(f"{dir}/*"):
        ext = os.path.splitext(os.path.basename(file))[1]
        if ext in patterns:
            img_path_list.append(file)
    img_filename_list = [os.path.splitext(os.path.basename(item))[0] for item in img_path_list]

    return img_path_list, img_filename_list

def walk_img_list(dir, ignore_dir_names, target_exts):
    img_path_list = []
    img_filename_list = []

    for root, dirs, files in os.walk(dir):
        if any(ignore_dir in dirs for ignore_dir in ignore_dir_names):
            dirs[:] = [d for d in dirs if d not in ignore_dir_names]

        for file in files:
            ext = os.path.splitext(file)[1].lower()
            
            if ext in target_exts:
                img_path = os.path.join(root, file)
                img_path_list.append(img_path)
                img_filename_list.append(os.path.splitext(file)[0])

    return img_path_list, img_filename_list


def calc_embedding(img_path, model: rt.InferenceSession):
    # load img from path
    img = Image.open(img_path)
    
    # get width and height
    height, width = model.get_inputs()[0].shape[1:3]

    # Get RGB Image from input
    img = img.convert("RGBA")
    new_img = Image.new("RGBA", img.size, "WHITE")
    new_img.paste(img, mask=img)
    img = new_img.convert("RGB")
    img = np.asarray(img)

    # RGB to BGR
    img = img[:,:,::-1]

    # Image Processing
    img = dbimutils.make_squire(img, height)
    img = dbimutils.smart_resize(img, height)
    img = img.astype(np.float32)
    img = np.expand_dims(img, 0)

    # get names
    input_name = model.get_inputs()[0].name
    label_name = model.get_outputs()[0].name

    outputs = model.run([label_name], {input_name: img})

    return outputs

def plot_matrix(vectors, filename_list):
    vectors_2d = np.vstack(vectors)
    similarity_matrix = cosine_similarity(vectors_2d)
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    dataframe = pd.DataFrame(similarity_matrix, columns=filename_list, index=filename_list)
    dataframe = dataframe.round(5)

    print(dataframe)


def plot_matrix_notebook(vectors, filepath_list, preview_img_size, src_urls=None):
    vectors_2d = np.vstack(vectors)
    similarity_matrix = cosine_similarity(vectors_2d)
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    filename_list = [os.path.basename(filepath) for filepath in filepath_list]
    dataframe = pd.DataFrame(similarity_matrix, columns=filename_list, index=filename_list)
    dataframe = dataframe.round(5)

    # add img to table
    max_img_size = preview_img_size
    for i, filepath in enumerate(filepath_list):
        image = Image.open(filepath)

        with io.BytesIO() as buffer:
            width, height = image.size

            if width > height:
                new_width = max_img_size
                new_height = int(height * (max_img_size / width))
            else:
                new_height = max_img_size
                new_width = int(width * (max_img_size / height))

            image = image.resize((new_width, new_height))
            image.save(buffer, format='png')
            img_bytes = buffer.getvalue()

        # table elements
        img_src = f"data:image/png;base64,{b64encode(img_bytes).decode()}"

        if src_urls != None:
            src_url = src_urls[i]

            ic(src_url)
            ic(filepath)
            dataframe.loc[os.path.basename(filepath), 'image'] = f'<a href="{src_url}"><img src="{img_src}" alt="image" width="{new_width}" height="{new_height}"></a>'

        else:
            ic(filepath)
            dataframe.loc[os.path.basename(filepath), 'image'] = f'<img src="{img_src}" alt="image" width="{new_width}" height="{new_height}">'

    # show
    display(HTML(dataframe.to_html(escape=False)))