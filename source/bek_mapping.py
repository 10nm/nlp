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
import os

# i = 0

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4096"

model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')







with open('wk-re.pkl', "rb") as fIn:
    stored_data = pickle.load(fIn)
    stored_sentences = stored_data['sentences']
    stored_embeddings = stored_data['embeddings']
    stored_ids = stored_data['id']
    print("file loaded")

    # Dumping
    with open('wktrc-re.pkl', 'wb') as f:
       emb = torch.stack(stored_embeddings)
       pickle.dump(emb, f)
       print("dumped!")

sentences = ['This framework generates embeddings for each input sentence',
             'Sentences are passed as a list of string.',
             'The quick brown fox jumps over the lazy dog.',
            ]

print(sentences)

sentences_embedding = model.encode(sentences)
print(sentences_embedding.shape)
print(sentences_embedding)

tsne = TSNE(n_components=2, perplexity=1)  # Decrease the value of perplexity

sentences_tsne = tsne.fit_transform(sentences_embedding)
sentences_tsne = sentences_tsne.tolist()
print(sentences_tsne)

x = []
y = []

def plot_sentence_embedding(sentences_tsne):
    for i in range(len(sentences_tsne)):
        print(sentences_tsne[i][0])
        print(sentences_tsne[i][1])

        x.append(sentences_tsne[i][0])
        y.append(sentences_tsne[i][1])
        
plot_sentence_embedding(sentences_tsne)

print(x)
print(y)

plt.scatter(x, y)
plt.savefig("map.png")
plt.show()

# cos_scores = util.cos_sim(query_embedding, sentences_embedding)[0]
# top_results = torch.topk(cos_scores, k=3)

# for score, idx in zip(top_results[0], top_results[1]):
#     print(idx)

#     print(sentences_embedding.shape)
#     sentences_embedding_reshaped = np.array(sentences_embedding[idx]).reshape(-1, 2)  # Reshape the array
#     print(sentences_embedding_reshaped.shape)

#     print(sentences_embedding_reshaped)
#     print(tsne.fit_transform(sentences_embedding_reshaped))
