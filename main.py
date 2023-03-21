## User Inputs

model_path = r"E:\AI\miatiadev\miatia-tagger\models\wd-v1-4-convnextv2-tagger-v2\model.onnx"
img_dir = r"E:\AI\miatiadev\miatia-tagger\imgs\00_similarity"

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

## User Inputs End

import onnxruntime as rt
import numpy as np
from Utils import inferences

from icecream import ic

# initialize
model = rt.InferenceSession(model_path,providers=providers)
if img_dir.endswith(("/", "\\")):
    img_dir = img_dir[:-1]

# get files
img_path_list, img_filename_list = inferences.get_img_list(img_dir)
embeddings = []
embedding_shapes = []
for i in range(len(img_path_list)):
    embedding = inferences.calc_embedding(img_path_list[i], model)
    embeddings.append(embedding)
    embedding_shapes.append(embedding.shape[1])

# print embeddings
for i in range(len(img_path_list)):
    ic(embeddings[i])
    ic(embeddings[i].shape[1])

# save embeddings
for i in range(len(img_path_list)):
    with open(img_dir + "/" + f'{img_filename_list[i]}.txt', 'w') as f:
        np.savetxt(f, embeddings[i])

#inferences.plot_matrix(embeddings, img_filename_list)

ic(embedding[0].ndim)

inferences.plot_matrix(embeddings, img_filename_list)