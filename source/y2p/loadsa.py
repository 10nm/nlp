# sentiment analysis with bert

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, BertJapaneseTokenizer
from transformers import TrainingArguments, Trainer

import seaborn as sns
import japanize_matplotlib

model = AutoModelForSequenceClassification.from_pretrained("koheiduck/bert-japanese-finetuned-sentiment")
tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# load dataset
from datasets import load_dataset

dataset = load_dataset("SetFit/amazon_reviews_multi_ja")
print(dataset)

# print(dataset['train']['text'][:10][:300])
# print(dataset['train']['text'][:-10][:300])
# print(dataset['train']['label'][:10])

scores = []

reviews = dataset['train']['text']
scores = dataset['train']['label']

elements = []  
stored_score = []


# # 星1
# st1 = dataset['train']['text'][:10]
# scores = dataset['train']['label'][:10]
i = 0
star = 0

for c , (review, score) in enumerate(zip(reviews,scores)):
    if score == star:
        for i in range(500):
            e = (reviews)[c+i][:250]
            s = (scores)[c+i]

            print(s)
            print(e)
            
            elements.append(e)
            stored_score.append(int(s+1))  # Convert scores to string before accessing as an indexable object
        print('---')
        # elements.append((reviews)[c:c+9])
        # stored_score.append((str(scores))[c:c+9])  # Convert scores to string before accessing as an indexable object
        star = star + 1



print(elements)
# stored_score = sum(stored_score, [])

print(stored_score)

# for value, i in zip(reversed(dataset['train']['text']), range(10)):
#     elements.append(value)

# for value, i in zip(reversed(dataset['train']['label']), range(10)):
#     scores.append(value)

# print(elements)
scaled_list = []
def scale_score(result):
    if result[0]['label'] == 'NEGATIVE':
        return result[0]['score'] * -1
    elif result[0]['label'] == 'POSITIVE':
        return result[0]['score']
    else:
        return 0
    
for i, element in enumerate(elements):
    before_scale = nlp(element)
    print(i)
    print(before_scale)
    scaled_list.append(scale_score(before_scale))
    # scaled_list.append(before_scale)

import itertools

#scaled_list = list(itertools.chain.from_iterable(scaled_list))
print(scaled_list)
print(stored_score)



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle

df = pd.DataFrame(
    {
        'scaled_list': scaled_list,
        'stored_score': stored_score
    }
)

df.to_pickle('scaled.pkl')
df = pd.read_pickle('scaled.pkl')

print(df)

sns.stripplot(x='stored_score', y='scaled_list', data=df, alpha=0.01, color='red')

# plt.scatter(scaled_list, stored_score, alpha=0.5)
# # sns.stripplot(x=scaled_list, y=stored_score, alpha=0.5, color='red')
# plt.ylabel('レビューのスコア')
# plt.xlabel('ネガティブ/ポジティブのスコア')
plt.yticks([-1, -0.5, 0 , 0.5, 1])


import datetime
savename = 'scatter_plot_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.png'
plt.savefig(savename)
plt.show()
