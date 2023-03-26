# sim-with-tagger  
指定フォルダ内の画像をベクトルにして比較して表にするアレの環境  

## Notebook  
3/23 追加    
[![](https://img.shields.io/static/v1?message=Open%20in%20Colab&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20&style=for-the-badge)](https://colab.research.google.com/github/miatia1025/sim-using-tagger/blob/main/sim_using_tagger.ipynb)  

## 環境(Win)  
powershell  
git  
python 3.10.6  

CUDA Toolkit 11.6  
cuDNN 8.5くらい  
-> https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements  

## 環境(WSL2)  
Ubuntu 22.0.4  
そのうち    

## 用意するもの  
こことかの  
https://huggingface.co/SmilingWolf/wd-v1-4-convnextv2-tagger-v2  
こういうの  
https://huggingface.co/SmilingWolf/wd-v1-4-convnextv2-tagger-v2/resolve/main/model.onnx  

あと数枚程度の画像が入ったフォルダ  

## 実行方法  
1/7 リポジトリのクローン  
`git clone https://github.com/miatia1025/sim-with-tagger`  
  
2/7 カレントディレクトリを移して  
`cd sim-with-tagger`  
  
3/7 venvを作って  
`py -3.10 -m venv venv`
  
4/7 activateして  
`./venv/scripts/activate`  
  
5/7 ライブラリをrequirementx.txtでインストールして  
`pip install -r requirements.txt`  

6/7 main.pyのmodel_path(taggerのモデルのパス), img_dir(比較したい画像の詰まったディレクトリ)を書き換える  
```py
model_path = r"path/to/tagger/model.onnx"
img_dir = r"path/to/images"
```  
具体的な例はこう  
```py
model_path = r"E:\AI\miatiadev\miatia-tagger\models\wd-v1-4-convnextv2-tagger-v2\model.onnx"
img_dir = r"E:\AI\miatiadev\miatia-tagger\imgs\00_similarity"
```  

7/7 実行  
`python ./main.py`
