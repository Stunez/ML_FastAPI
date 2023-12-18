import asyncpg
import asyncio
import json
import pandas as pd

from fastapi import Depends, FastAPI, Response, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from config import *
from modules.models import *
from modules.qa_processing import rb_model_answer, bb_model_answer
from modules.DBInteraction import DBInteraction


# application instance
app = FastAPI(
    redoc_url="/api/v0.1/developer",
    title="Question-Answer JeopardyDataset API",
    version="0.1",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
)
# middleware for sending gzip response
app.add_middleware(GZipMiddleware)
# CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB = DBInteraction(DB_NAME, SCHEMA, ENGINE)

# connector to DB
async def connect_to_db():
    # max_size, min_size - количество коннекшнов
    conn = await asyncpg.create_pool(dsn=DATABASE_URL,
                                     min_size=5, max_size=50)
    return conn


@app.on_event("startup")
async def startup():
    app.state.conn = await connect_to_db()
    
    
@app.on_event("shutdown")
async def shutdown():
    await app.state.conn.close()


@app.get("/", include_in_schema=False)
async def docs_redirect():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="Question-Answer JeopardyDataset API")


@app.post(
    path='/api/question/bert', 
    response_model=QuestionOutput,
    tags=['Вопрос-ответ'],
    summary='Получение ответа по модели "RoBERTa-Base"',
    responses= {200: {"content": {"application/json" : {}}}},
    description="""
        Модель, натренированная на датасете 'JEOPARDY_CSV' для 
        предугадывания ответов, была обучена поверх модели 'RoBERTa-Base'.
    """
)
async def get_rb_model_answer(input : QuestionInput):
        
    # Running function as asynchronous by asyncio
    response = await asyncio.to_thread(rb_model_answer.output, input.question)
    await DB.df_build('RoBERTa-Base', input.question, response)
    response = json.dumps(response, ensure_ascii=False, default=str)
    return Response(content=response, media_type='application/json')


@app.post(
    path='/api/question/robert', 
    response_model=QuestionOutput,
    tags=['Вопрос-ответ'],
    summary='Получение ответа по модели "BERT-Base-Uncased"',
    responses= {200: {"content": {"application/json" : {}}}},
    description="""
        Модель, натренированная на датасете 'JEOPARDY_CSV' для 
        предугадывания ответов, была обучена поверх модели 'BERT-Base-Uncased'.
    """
)
async def get_bb_model_answer(input : QuestionInput):
    # Running function as asynchronous by asyncio
    response = await asyncio.to_thread(bb_model_answer.output, input.question)
    # response = await bb_model_answer.output(input.question)
    await DB.df_build('BERT-Base-Uncased', input.question, response)
    response = json.dumps(response, ensure_ascii=False, default=str)
    return Response(content=response, media_type='application/json')


@app.get(
    path='/api/data', 
    tags=['Данные Вопрос-ответной истории'],
    response_model=QAOutput,
    summary='Получение таблички и ее данные',
    responses= {200: {"content": {"application/json" : {}}}},
    description="Получение истории по пользовательским запросам"
)
async def get_qa_data():
    response = await app.state.conn.fetch(qa_data_query)  
    response = json.dumps(response, ensure_ascii=False, default=str)
    return Response(content=response, media_type='application/json')