from django.contrib import admin, messages
from .models import *
from .forms import CsvUploadForm
from rangefilter.filters import DateRangeFilter

import pandas as pd
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

import os

import openai
# Register your models here.

@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ('user_id', 'email', 'is_active', 'is_admin', 'created_at', 'updated_at')
    search_fields = ('user_id', 'email')
    list_filter = ('is_active', 'is_admin', 'created_at')
    ordering = ('user_id',)

@admin.register(Chat)
class ChatAdmin(admin.ModelAdmin):
    list_display = ('chat_id', 'chat_name', 'user', 'created_at', 'updated_at')
    search_fields = ('chat_id', 'chat_name', 'user__user_id')
    list_filter = ('created_at', 'updated_at')

@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ('chat', 'sender', 'content', 'timestamp')
    search_fields = ('chat__chat_id', 'sender', 'content')
    list_filter = ('timestamp',)

@admin.register(AivleChatHistory)
class AivleChatHistoryAdmin(admin.ModelAdmin):
    list_display = ('datetime', 'query', 'sim1', 'sim2', 'sim3', 'answer')
    search_fields = ('query', 'answer')
    list_filter = (
        ('datetime', DateRangeFilter),
    )

class CsvUploadAdmin(admin.ModelAdmin):
    form = CsvUploadForm

    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)
        # CSV 파일을 읽고 처리하는 로직 추가
        file_path = obj.file.path
        data = pd.read_csv(file_path, encoding='utf-8')

        # OpenAI의 "text-embedding-ada-002" 모델을 사용하여 텍스트 임베딩 객체를 생성
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

        # Chroma 데이터베이스 인스턴스를 생성하고, 임베딩 함수를 설정
        database = Chroma(persist_directory="./QnA_DB", embedding_function=embeddings)

        # 데이터프레임의 '질문', '답변', '분류' 열에서 텍스트를 가져와 Document 객체의 리스트로 변환
        documents = [Document(page_content=f"질문: {row['질문']} 답변: {row['답변']} 분류: {row['분류']}") for _, row in data.iterrows()]

        # 생성한 Document 객체들을 Chroma 데이터베이스에 추가
        database.add_documents(documents)

admin.site.register(CsvUpload, CsvUploadAdmin)