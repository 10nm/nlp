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
import os

i = 0

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4096"

model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')

with open('wk-re-rm.pkl', "rb") as fIn:
    stored_data = pickle.load(fIn)
    # stored_sentences = stored_data['sentences']
    stored_embeddings = stored_data['embeddings']
    stored_embeddings = torch.stack(stored_embeddings)
    stored_ids = stored_data['id']
    print("file loaded")


#     ## Dumping
#     # with open('wktrc-re.pkl', 'wb') as f:
#     #     emb = torch.stack(stored_embeddings)
#     #     pickle.dump(emb, f)
#     #     print("dumped!")

query="自然言語処理"
query_embedding = model.encode(query, convert_to_tensor=True)
print("query embedded")

## Dump load
# with open('wktrc-re.pkl', 'rb') as f:
#     emb = pickle.load(f)
#     print("emb loaded")

cos_scores = util.cos_sim(query_embedding, stored_embeddings)[0]
top_results = torch.topk(cos_scores, k=30)

topemb = []
topids = []
topscores = []

for score, idx in zip(top_results[0], top_results[1]):
    topemb.append(stored_embeddings[idx])
    topids.append(stored_ids[idx])
    topscores.append(score)

    i += 1
    print(str(i), ":\n")

    print("score: ")
    print(score, "\n")

    print("id: ")
    print(stored_ids[idx], "\n")

    print("wiki page:")
    print("https://ja.wikipedia.org/w/index.php?curid=" + str(stored_ids[idx]))

topemb = torch.stack(topemb)


with open('topemb.pkl', 'wb') as f:
    pickle.dump({"embeddings": topemb, "id": topids, "score": topscores}, f)
