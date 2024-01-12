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
import matplotlib.pyplot as plt
import japanize_matplotlib
import os
from adjustText import adjust_text

# i = 0

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4096"

with open('topemb.pkl', "rb") as fIn:
    stored_data = pickle.load(fIn)
    stored_embeddings = stored_data['embeddings'].cpu().numpy()
    stored_ids = stored_data['score']
    stored_titles = stored_data['title']
    stored_scores = stored_data['score']
    print("file loaded")

tsne = TSNE(n_components=2, perplexity=2)  # Decrease the value of perplexity

sentences_tsne = tsne.fit_transform(stored_embeddings)
sentences_tsne = sentences_tsne.tolist()
print(sentences_tsne)


x = []
y = []
titles=[]

def plot_sentence_embedding(sentences_tsne):
    for i in range(len(sentences_tsne)):
        print(sentences_tsne[i][0])
        print(sentences_tsne[i][1])

        x.append(sentences_tsne[i][0])
        y.append(sentences_tsne[i][1])
        titles.append(str(stored_titles[i]))

plot_sentence_embedding(sentences_tsne)

X = list(x)
Y = list(y)
Title = list(titles)
Score = list(stored_scores)
print(Title)
print(Score)
i = 0

plt.figure()

#title
plt.title("「自然言語処理」の検索結果上位30記事", fontsize=15)

plt.scatter(X, Y)

texts = [plt.text(X[i], Y[i], label) for i, label in enumerate(Title)]
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))

plt.savefig("map.png")
plt.show(block=True)


# cos_scores = util.cos_sim(query_embedding, sentences_embedding)[0]
# top_results = torch.topk(cos_scores, k=3)

# for score, idx in zip(top_results[0], top_results[1]):
#     print(idx)

#     print(sentences_embedding.shape)
#     sentences_embedding_reshaped = np.array(sentences_embedding[idx]).reshape(-1, 2)  # Reshape the array
#     print(sentences_embedding_reshaped.shape)

#     print(sentences_embedding_reshaped)
#     print(tsne.fit_transform(sentences_embedding_reshaped))
