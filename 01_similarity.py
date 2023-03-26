## User Inputs

model_path = r"E:\AI\miatiadev\miatia-tagger\models\wd-v1-4-swinv2-tagger-v2\model.onnx"
img_dir = r"E:\AI\miatiadev\miatia-tagger\imgs\03_test200tags"
csv_path = r"E:\AI\miatiadev\miatia-tagger\models\wd-v1-4-convnextv2-tagger-v2\selected_tags.csv"

general_threshold = 0.35
character_threshold = 0.35

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

## User Inputs End

import onnxruntime as rt
import numpy as np
import os
import time

from Utils import inferences
from icecream import ic

# initialize
init_time = time.time()

model = rt.InferenceSession(model_path,providers=providers)
if img_dir.endswith(("/", "\\")):
    img_dir = img_dir[:-1]
os.makedirs(f"{img_dir}/npy", exist_ok=True)

# get labels
tag_names, rating_indexes, general_indexes, character_indexes = inferences.get_labels(csv_path)


# get files
img_path_list, img_filename_list = inferences.get_img_list(img_dir)

embeddings = []
for i in range(len(img_path_list)):

    emb_txt = img_dir + "/npy/" + f"{img_filename_list[i]}.npy"
    
    if os.path.exists(emb_txt):
      probs = np.load(emb_txt)
      embedding = probs[0]
      ic(embedding)

    else:
      probs = inferences.calc_embedding(img_path_list[i], model)
      embedding = probs[0]
      np.save(emb_txt, probs)

    embeddings.append(embedding)

# print embeddings
for i in range(len(img_path_list)):
    ic(embeddings[i])
    ic(embeddings[i].shape[1])


inferences.plot_matrix(embeddings, img_filename_list)

term_time = time.time()
total_time = term_time - init_time
print("Total time: {:.3f} seconds".format(total_time))