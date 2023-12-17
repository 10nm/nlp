import random
import json
import glob
from tqdm import tqdm
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertJapaneseTokenizer, BertForMaskedLM
from sentence_transformers import SentenceTransformer, util
import re
import pickle
import math

model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')

sfile = "wkt.json"
mx = 1386531

def save_pkl(tx,vc,aid):
    with open('wk.pkl', "wb") as vec:
        pickle.dump({"sentences": tx, "embeddings": vc, "id": aid}, vec, protocol=pickle.HIGHEST_PROTOCOL) 

vectors = []
texts=[]
ids=[]
i=0

#memo
# https://ja.wikipedia.org/w/index.php?curid= ここにidが入る

#max_learn = 1000000

with open(sfile, "r",  encoding="utf-8") as f:
    for line in f:
        if i % 1000 == 0:
            print(i, "/", (round(i/mx, 3)*100), "%")

        # load json
        json_load = json.loads(line)
        # 記事本文
        tex = json_load["text"]
        # 記事id
        artid = json_load["id"]

        text = tex[:700]
        raw_text = tex

        ## URLを削除
        #rmurl = re.sub("https?://[\w/:%#\$&\?\(\)~\.=\+\-]+", "", tex)

        ## かっこを削除
        #rmbr = re.sub('\(.*?\)', "", rmurl)
        #rmbr = re.sub('（.*?）', "", rmbr)

        ## ^ から始まる文字列を削除
        #tex = re.sub("\^ [a-z ]*", "", rmbr)

        ##年月日の削除
        #text = re.sub("\d{4}[/\.年]\d{1,2}[/\.月]\d{1,2}日?", "", tex)

        vec = model.encode(text, convert_to_tensor=True, device="cuda")
        
        vectors.append(vec)  # ベクトルをリストに追加
        texts.append(text)  # テキストをリストに追加
        ids.append(artid)
        
        #if i == max_learn:
        #    break
        i += 1
#save pkl
save_pkl(texts,vectors,ids)
print("saved")