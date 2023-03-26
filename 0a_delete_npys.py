## User Inputs

model_path = r"E:\AI\miatiadev\miatia-tagger\models\wd-v1-4-convnextv2-tagger-v2\model.onnx"
img_dir = r"E:\AI\miatiadev\miatia-tagger\imgs\04_recuisive"
csv_path = r"E:\AI\miatiadev\miatia-tagger\models\wd-v1-4-convnextv2-tagger-v2\selected_tags.csv"

ignore_dir_names = ("json", "npy", )
target_exts = (".png", ".jpg", ".jpeg", ".PNG", ".JPEG", ".JPG", )

general_threshold = 0.1
character_threshold = 0.1

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

## User Inputs End

import os

from Utils import inferences
from icecream import ic

# initialize
if img_dir.endswith(("/", "\\")):
    img_dir = img_dir[:-1]

# get files
img_path_list, img_filename_list = inferences.walk_img_list(img_dir, ignore_dir_names, target_exts)


for i in range(len(img_path_list)):
    emb_npy = os.path.join(os.path.dirname(img_path_list[i]), "npy", f"{img_filename_list[i]}.npy")
    tag_txt = os.path.join(os.path.dirname(img_path_list[i]), f"{img_filename_list[i]}.txt")

    #ic(emb_npy)
    #ic(tag_txt)
    
    if os.path.exists(emb_npy) and not any(name in emb_npy for name in ignore_dir_names):
        os.remove(emb_npy)
        print(f"removed {img_filename_list[i]}.npy")
        pass

    if os.path.exists(tag_txt):
        os.remove(tag_txt)
        print(f"removed {img_filename_list[i]}.txt")
        pass
