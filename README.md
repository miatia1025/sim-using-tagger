# sim-with-tagger  
指定フォルダ内の画像をベクトルにして比較して表にするアレの環境  
  
## 環境  
powershell  
git  
python 3.10.6  

CUDA Toolkit 11.6
cuDNN 8.

## 実行方法  
1/7 リポジトリのクローン  
`git clone https://github.com/miatia1025/sim-with-tagger`  
  
2/7 カレントディレクトリを移して  
`cd sim-with-tagger`  
  
3/7 venvを作って  
`py -3.10 -m venv venv`
  
4/7 activateして  
`./venv/scripts/activate`  
  
5/7 ライブラリをインストールして
`pip install -r requirements`  

6/7 main.pyのmodel_path(taggerのモデルのパス), img_dir(比較したい画像の詰まったディレクトリ)を書き換える  
```py
model_path = r"path/to/tagger/model.onnx"
img_dir = r"path/to/images"
```  
具体的にはこう  
```py
model_path = r"E:\AI\miatiadev\miatia-tagger\models\wd-v1-4-convnextv2-tagger-v2\model.onnx"
img_dir = r"E:\AI\miatiadev\miatia-tagger\imgs\00_similarity"
```  

7/7 実行  
`python ./main.py`
