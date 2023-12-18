import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import datetime as dt

# Проверка доступности GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "distilbert-base-uncased"

# Класс для датасета
class QuestionsAnswersDataset(Dataset):
    def __init__(self, questions, answers, tokenizer, max_len):
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, item):
        question = str(self.questions[item])
        answer = self.answers[item]

        encoding = self.tokenizer.encode_plus(
            question,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(answer, dtype=torch.long)
        }

# Функция для обучения
def train_epoch(model, data_loader, optimizer, device):
    model = model.train()
    losses = []

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return sum(losses) / len(losses)

# Функция для предсказания
def predict(model, question, tokenizer, max_len):
    encoding = tokenizer.encode_plus(
        question,
        add_special_tokens=True,
        max_length=max_len,
        return_attention_mask=True,
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        model.eval()
        outputs = model(input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    return prediction


def main():
    
    # Загрузка и подготовка данных
    data = pd.read_csv('data/jeopardy_renewed.csv')
    data.columns = [x.strip().lower() for x in data.columns.to_list()]
    data = data.dropna(subset=['question','answer'])
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    max_len = 512

    # Преобразование ответов в числовые метки
    label_encoder = LabelEncoder()
    data['encoded_answers'] = label_encoder.fit_transform(data['answer'])

    # Разделение данных на обучающую и тестовую выборки
    train_data, test_data = train_test_split(data, test_size=0.1)

    train_dataset = QuestionsAnswersDataset(
        questions=train_data.question.to_numpy(),
        answers=train_data.encoded_answers.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    train_data_loader = DataLoader(train_dataset, batch_size=16)

    # Загрузка и настройка модели
    model = DistilBertForSequenceClassification.from_pretrained( 
        model_name,
        num_labels=len(label_encoder.classes_)
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    started = dt.datetime.now()

    # Обучение модели
    for epoch in range(3):  # количество эпох
        # 1 epoch take 1 hour - 57 min 35 sec
        avg_loss = train_epoch(model, train_data_loader, optimizer, device)
        print(f"Эпоха {epoch + 1}, Средний Loss: {avg_loss}")

    # Сохранение модели
    model.save_pretrained('data/models/jeopardy_' + model_name)

    print(dt.datetime.now() - started)
    
if __name__ == '__main__':
    main()