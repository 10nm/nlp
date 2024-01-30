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

elements = dataset['train']['text'][:10]
scores = dataset['train']['label'][:10]

for value, i in zip(reversed(dataset['train']['text']), range(10)):
    elements.append(value)

for value, i in zip(reversed(dataset['train']['label']), range(10)):
    scores.append(value)

print(elements)

scaled_list = []

def scale_score(result):
    if result[0]['label'] == 'NEGATIVE':
        return result[0]['score'] * -1
    elif result[0]['label'] == 'POSITIVE':
        return result[0]['score']
    else:
        return result[0]['score']
    
for element in elements:
    before_scale = nlp(element)
    scaled_list.append(scale_score(before_scale))

print(scaled_list)

scaled_to_score = []



print(scores)

