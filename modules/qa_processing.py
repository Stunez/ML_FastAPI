import asyncio

import pandas as pd
import torch

from sklearn.preprocessing import LabelEncoder
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification, 
    BertTokenizer, BertForSequenceClassification
)


# Загрузка и подготовка данных
data = pd.read_csv('data/jeopardy_renewed.csv')
data.columns = [x.strip().lower() for x in data.columns.to_list()]
data = data.dropna(subset=['question','answer'])

# Преобразование ответов в числовые метки
label_encoder = LabelEncoder()
data['encoded_answers'] = label_encoder.fit_transform(data['answer'])


class Answer():
    
    def __init__(self, model_name_base, Tokenizer, sequence_classification_object):
        # Проверка доступности GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = Tokenizer.from_pretrained(model_name_base)
        self.max_len = 256 # max 512 for bert and robert
        self.model = sequence_classification_object.from_pretrained('data/models/jeopardy_' + model_name_base).to(self.device)
    
    
    def predict(self, model, question, tokenizer, max_len):
        """ Функция для предсказания."""
        encoding = tokenizer.encode_plus(
            question,
            add_special_tokens=True,
            max_length=max_len,
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            model.eval()
            outputs = model(input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(outputs.logits, dim=1).item()

        return prediction
    
    
    def output(self, question):
        
        # Running function as asynchronous by asyncio
        # prediction =  await asyncio.to_thread(self.predict, self.model, question, self.tokenizer, self.max_len)
        prediction =  self.predict(self.model, question, self.tokenizer, self.max_len)
        predicted_answer = label_encoder.inverse_transform([prediction])[0]
        
        return predicted_answer


rb_model_answer = Answer('roberta-base', RobertaTokenizer, RobertaForSequenceClassification)
bb_model_answer = Answer('bert-base-uncased', BertTokenizer, BertForSequenceClassification)
