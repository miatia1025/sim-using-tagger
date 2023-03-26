## User Inputs

model_path = r"E:\AI\miatiadev\miatia-tagger\models\wd-v1-4-swinv2-tagger-v2\model.onnx"
img_dir = r"E:\AI\miatiadev\miatia-tagger\imgs\resized_pixiv_filtered_omit_pixiv_48456_and_low-med-res"
csv_path = r"E:\AI\miatiadev\miatia-tagger\models\wd-v1-4-convnextv2-tagger-v2\selected_tags.csv"

ignore_dir_names = ("npy")
target_exts = (".png", ".jpg", ".jpeg", ".PNG", ".JPEG", ".JPG")

general_threshold = 0.1
character_threshold = 0.1

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

'''
if img_dir.endswith(("/", "\\")):
    img_dir = img_dir[:-1]
os.makedirs(f"{img_dir}/npy", exist_ok=True)
'''

# get labels
tag_names, rating_indexes, general_indexes, character_indexes = inferences.get_labels(csv_path)

# get files
img_path_list, img_filename_list = inferences.walk_img_list(img_dir, ignore_dir_names, target_exts)

for item in img_path_list:
  if item.endswith(("/", "\\")):
    item = item[:-1]
    
  os.makedirs(f"{os.path.dirname(item)}/npy", exist_ok=True)
  

for i in range(len(img_path_list)):
    start_time = time.time()

    emb_txt = os.path.dirname(img_path_list[i]) + "/npy/" + f"{img_filename_list[i]}.npy"
    ic(emb_txt)
    
    if os.path.exists(emb_txt):
      probs = np.load(emb_txt)
      embedding = probs[0]
      #ic(embedding)

    else:
      probs = inferences.calc_embedding(img_path_list[i], model)
      embedding = probs[0]
      np.save(emb_txt, probs)
    
    # scores
    tag_strings, raw_tag_strings, rating_result, general_result, character_result = inferences.get_scores(embedding, general_threshold, character_threshold, tag_names, rating_indexes, general_indexes, character_indexes)


    # store embeddings
    #ic(probs)


    # save tag list
    np.savetxt(os.path.dirname(img_path_list[i]) + "/" + f"{img_filename_list[i]}.txt", [tag_strings], fmt="%s")

    # for development
    end_time = time.time()
    elapced_time = end_time - start_time
    print("Elapced time: {:.3f} seconds".format(elapced_time))


# print embeddings
'''
for i in range(len(img_path_list)):
    ic(embeddings[i])
    ic(embeddings[i].shape[1])
'''

# inferences.plot_matrix(embeddings, img_filename_list)

term_time = time.time()
total_time = term_time - init_time
print("Total time: {:.3f} seconds".format(total_time))