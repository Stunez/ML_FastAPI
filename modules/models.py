from pydantic import BaseModel, Field
from typing import Union


class QuestionInput(BaseModel):
    question: Union[str, None] = Field(
        description='Вопросительное предложение')  
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "Which city is capital of France?"
            },
        }
        
        
class QuestionOutput(BaseModel):
    answer: Union[str, None] = Field(
        description='Ответ на заданный вопрос пользователем') 
    														
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Paris"
            },
        }


class QAOutput(BaseModel):
    
    id: Union[int, None] = Field(
        description='Идентификационный номер')
    my_model_name: Union[str, None] = Field(
        description='Название модели') 
    question: Union[str, None] = Field(
        description='Вопросительное предложение') 
    answer: Union[str, None] = Field(
        description='Ответ на вопросительное предложение') 
    create_date: Union[str, None] = Field(
        description='Дата и время создания запроса') 
    														
    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "model_name": "RoBERTa-Base",
                'question': "Which city is capital of France?",
                'answer': 'Paris',
                'create_date': '2023-12-18 05:16:06.000'
            },
        }