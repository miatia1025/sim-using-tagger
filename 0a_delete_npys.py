## User Inputs

model_path = r"E:\AI\miatiadev\miatia-tagger\models\wd-v1-4-convnextv2-tagger-v2\model.onnx"
img_dir = r"E:\AI\miatiadev\miatia-tagger\imgs\03_test200tags"
csv_path = r"E:\AI\miatiadev\miatia-tagger\models\wd-v1-4-convnextv2-tagger-v2\selected_tags.csv"

general_threshold = 0.35
character_threshold = 0.35

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

## User Inputs End

import os

from Utils import inferences

# initialize
if img_dir.endswith(("/", "\\")):
    img_dir = img_dir[:-1]

# get files
img_path_list, img_filename_list = inferences.get_img_list(img_dir)
for i in range(len(img_path_list)):
    emb_npy = img_dir + "/npy/" + f"{img_filename_list[i]}.npy"
    tag_txt = img_dir + "/" + f"{img_filename_list[i]}.txt"
    
    if os.path.exists(emb_npy):
        os.remove(emb_npy)
        print(f"removed {img_filename_list[i]}.npy")

    if os.path.exists(tag_txt):
        os.remove(tag_txt)
        print(f"removed {img_filename_list[i]}.txt")


    else:
        pass
