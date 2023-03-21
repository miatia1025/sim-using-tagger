import onnxruntime as rt
import numpy as np
import os
import glob
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from Utils import dbimutils


def get_img_list(dir):
    img_path_list = []
    patterns = (".png", ".jpg", ".jpeg", ".PNG", ".JPEG", ".JPG")

    for file in glob.glob(f"{dir}/*"):
        ext = os.path.splitext(os.path.basename(file))[1]
        if ext in patterns:
            img_path_list.append(file)
    img_filename_list = [os.path.splitext(os.path.basename(item))[0] for item in img_path_list]

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

    outputs = model.run(None, {input_name: img})
    embedding = outputs[0]

    return embedding

def plot_matrix(vectors, filename_list):
    vectors_2d = np.vstack(vectors)
    similarity_matrix = cosine_similarity(vectors_2d)

    dataframe = pd.DataFrame(similarity_matrix, columns=filename_list, index=filename_list)
    dataframe = dataframe.round(5)

    print(dataframe)

    return 0