# views.py
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.http import JsonResponse
from django import forms
from django.urls import reverse
from django.utils import timezone
from django.contrib.sessions.models import Session
from django.db import transaction


# from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory , ChatMessageHistory 
from langchain.schema import Document

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI

from .models import *

import pandas as pd
import json
import os

import openai

import uuid

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Create your views here.
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from .forms import LoginForm

# views.py
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from .forms import *

from django.utils.dateparse import parse_date
from django.db.models import Q

# 유저 객체
User = get_user_model()

embeddings = OpenAIEmbeddings(model = "text-embedding-ada-002")
database = Chroma(persist_directory = "./QnA_DB", embedding_function = embeddings)



# HomePage
def home(request):
    return render(request, 'ChatSystem/home/home.html')

@login_required
def nav(request):
    return render(request, 'ChatSystem/home/nav.html')

# Signup
def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        #print(form)
        if form.is_valid():
            print(form.cleaned_data['user_id'])
            user = form.save()
            login(request, user)  # 회원가입 후 자동 로그인
            return redirect('ChatSystem:nav')
        else:
            print(form.errors)
    else:
        form = SignUpForm()
    return render(request, 'ChatSystem/sign/signup.html', {'form': form})

# Login
def user_login(request):
    # 이미 로그인된 사용자인지 확인
    if request.user.is_authenticated:
        return redirect('ChatSystem:nav')
    
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            user_id = form.cleaned_data['user_id']
            password = form.cleaned_data['password']
            user = authenticate(request, username=user_id, password=password)
            if user is not None:
                login(request, user)
                # 세션에 사용자 정보를 저장
                request.session['user_id'] = user.user_id
                request.session['email'] = user.email
                return redirect('ChatSystem:nav')
            else:
                return HttpResponse("로그인 정보가 올바르지 않습니다.")
    else:
        form = LoginForm()

    return render(request, 'ChatSystem/sign/login.html', {'form': form})

# Logout
@login_required
def user_logout(request):
    # 세션 데이터 삭제
    if 'user_id' in request.session:
        del request.session['user_id']
    if 'email' in request.session:
        del request.session['email']
    
    logout(request)
    return redirect('ChatSystem:login')

# user_data
@login_required
def user_info(request):
    user = request.user
    if request.method == 'POST':
        form = UserUpdateForm(request.POST, instance=user)
        if form.is_valid():
            form.save()
            return redirect('ChatSystem:user_info')
    else:
        form = UserUpdateForm(instance=user)

    context = {
        'user_id': user.user_id,
        'form': form,
    }

    return render(request, 'ChatSystem/sign/user_info.html', context=context)

# user_delete
@login_required
def delete_user(request):
    user = request.user
    user.delete()
    logout(request)
    return redirect('ChatSystem:login')  # 삭제 후 로그인 페이지로 리디렉션

####### chat #######

# HumanMessage 객체를 JSON으로 직렬화하는 함수
def serialize_memory(memory):
    return json.dumps([{"type": "human", "data": msg.content} if isinstance(msg, HumanMessage) 
                       else {"type": "ai", "data": msg.content} for msg in memory])

def deserialize_memory(serialized_memory):
    memory = []
    
    # serialized_memory가 딕셔너리인지 확인
    if isinstance(serialized_memory, str):
        try:
            loaded_memory = json.loads(serialized_memory)
        except json.JSONDecodeError:
            raise TypeError("Invalid JSON format")
    else:
        loaded_memory = serialized_memory

    # loaded_memory가 리스트인지 확인
    if isinstance(loaded_memory, list):
        for msg in loaded_memory:
            if isinstance(msg, dict):
                if msg.get("type") == "human":
                    memory.append(HumanMessage(content=msg.get("data", "")))
                else:
                    memory.append(AIMessage(content=msg.get("data", "")))
    else:
        raise TypeError("Deserialized memory should be a list of messages")

    return memory


# Chat index
@login_required
def index_chat(request):
    user = request.user
    if 'current_chat_id' in request.session:
        chat_id = request.session['current_chat_id']
    else:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        chat_id = str(uuid.uuid4())
        Chat.objects.create(chat_id=chat_id, user=user, chat_name="ChatGPT")
        request.session['chat_memory'] = {chat_id: serialize_memory(memory.chat_memory.messages)}
        request.session['current_chat_id'] = chat_id
    return render(request, 'ChatSystem/chat/index.html', {'chat_id': chat_id})

# Chat GPT ajax
@login_required
def chat_ajax(request):
    if request.method == 'POST':
        query = request.POST.get('question')
        chat_id = request.POST.get('chat_id')
        user = request.user

        chat_memory = request.session.get('chat_memory', {})
        messages = deserialize_memory(chat_memory.get(chat_id, "[]"))

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        memory.chat_memory.messages = messages

        memory.chat_memory.add_user_message(query)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant."}] +
                     [{"role": "user" if isinstance(msg, HumanMessage) else "assistant", "content": msg.content} for msg in memory.chat_memory.messages]
        )

        answer = response.choices[0].message["content"]

        memory.chat_memory.add_ai_message(answer)

        chat_memory[chat_id] = serialize_memory(memory.chat_memory.messages)
        request.session['chat_memory'] = chat_memory

        chat = Chat.objects.get(chat_id=chat_id, user=user, chat_name="ChatGPT")
        Message.objects.create(chat=chat, sender='human', content=query)
        Message.objects.create(chat=chat, sender='ai', content=answer)
        chat.save()

        response_data = {
            'question': query,
            'result': answer,
            'chat_id': chat_id,
        }

        return JsonResponse(response_data)
    else:
        return render(request, 'ChatSystem/chat/index.html')

# Aivle Chat index
@login_required
def index_aivle_chat(request):
    user = request.user
    if 'current_chat_id' in request.session:
        chat_id = request.session['current_chat_id']
    else:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        chat_id = str(uuid.uuid4())
        Chat.objects.create(chat_id=chat_id, user=user, chat_name="AivleChat")
        request.session['chat_memory'] = {chat_id: serialize_memory(memory.chat_memory.messages)}
        request.session['current_chat_id'] = chat_id
    return render(request, 'ChatSystem/aivle_chat/index.html', {'chat_id': chat_id})

# Aivle Chat ajax
@login_required
def aivle_chat_ajax(request):
    if request.method == 'POST':
        query = request.POST.get('question')
        chat_id = request.POST.get('chat_id')
        user = request.user

        # 세션에서 대화 메모리를 불러오기
        chat_memory = request.session.get('chat_memory', {})
        messages = deserialize_memory(chat_memory.get(chat_id, "[]"))

        # 대화 메모리 로드
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        memory.chat_memory.messages = messages

        # ChatGPT API 및 langchain 사용을 위한 선언
        chat = ChatOpenAI(model="gpt-3.5-turbo")
        k = 3  # 검색 결과 개수 설정
        retriever = database.as_retriever(search_kwargs={"k": k})  # database 객체는 사전에 정의되어 있어야 합니다.
        qa = ConversationalRetrievalChain.from_llm(llm=chat, retriever=retriever, memory=memory)

        # 유사도 점수
        sim_result = database.similarity_search_with_score(query, k = k) #← 데이터베이스에서 유사도가 높은 문서를 가져옴
        sim1 = round(sim_result[0][1], 5)
        sim2 = round(sim_result[1][1], 5)
        sim3 = round(sim_result[2][1], 5)

        # 현재 질문에 대한 응답 생성
        result = qa({"question": query, "chat_history": memory.load_memory_variables({})["chat_history"]})

        # 대화 메모리에 현재 질문과 응답 추가
        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(result["answer"])

        # 대화 메모리를 세션에 다시 저장
        chat_memory[chat_id] = serialize_memory(memory.chat_memory.messages)
        request.session['chat_memory'] = chat_memory

        # 데이터베이스에 메시지 저장
        chat = Chat.objects.get(chat_id=chat_id, user=user, chat_name="AivleChat")
        Message.objects.create(chat=chat, sender='human', content=query)
        Message.objects.create(chat=chat, sender='ai', content=result["answer"])
        chat.save()

        AivleChatHistory.objects.create(query=query, sim1=sim1, sim2=sim2, sim3=sim3, answer=result["answer"])

        # JSON 응답 반환
        response_data = {
            'question': query,
            'result': result["answer"],
            'chat_id': chat_id,
        }

        return JsonResponse(response_data)
    else:
        return render(request, 'ChatSystem/chat/index.html')


# Load chat history
@login_required
def load_chat_history(request):
    chat_id = request.GET.get('chat_id')
    user = request.user

    if Chat.objects.filter(chat_id=chat_id, user=user).exists():
        chat = Chat.objects.get(chat_id=chat_id, user=user)
        messages = chat.messages.all()
        response_data = {
            'messages': [{"type": "human", "data": msg.content} if msg.sender == 'human' 
                         else {"type": "ai", "data": msg.content} for msg in messages]
        }
        return JsonResponse(response_data)
    else:
        return JsonResponse({'messages': []})

# Load chat list
@login_required
def load_chat_list(request):
    user = request.user
    chats = Chat.objects.filter(user=user).order_by('-updated_at')
    response_data = {
        'chats': [{"chat_id": chat.chat_id, "chat_name" : chat.chat_name} for chat in chats]
    }
    return JsonResponse(response_data)

# 대화목록 생성
@login_required
def create_chat(request):
    if request.method == 'POST':
        user = request.user
        chat_id = str(uuid.uuid4())
        chat_name = request.POST.get('chat_name', 'ChatGPT')
        Chat.objects.create(chat_id=chat_id, user=user, chat_name=chat_name)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        request.session['chat_memory'] = {chat_id: serialize_memory(memory.chat_memory.messages)}
        request.session['current_chat_id'] = chat_id
        return JsonResponse({'chat_id': chat_id})
    return JsonResponse({'status': 'error', 'message': 'Invalid request method.'}, status=400)

# 대화목록 삭제
@login_required
def delete_chat(request, chat_id):
    try:
        with transaction.atomic():
            # 채팅 객체를 가져옵니다.
            chat = Chat.objects.get(chat_id=chat_id, user=request.user)
            chat.delete()
            
            # 세션에서 chat_id를 삭제합니다.
            chat_memory = request.session.get('chat_memory', {})
            if chat_id in chat_memory:
                del chat_memory[chat_id]
                request.session['chat_memory'] = chat_memory
            else:
                # chat_id가 세션에 없을 경우 경고 메시지 반환
                return JsonResponse({'status': 'error', 'message': 'Chat memory not found in session'}, status=400)
            
            # 현재 채팅 ID가 삭제된 채팅 ID와 같은 경우 세션에서 제거.
            if request.session.get('current_chat_id') == chat_id:
                del request.session['current_chat_id']
            
            # 세션 데이터를 명시적으로 저장합니다.
            request.session.modified = True

            return JsonResponse({'status': 'success', 'message': 'Chat deleted successfully'})
    except Chat.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Chat not found'}, status=404)
    except KeyError as e:
        return JsonResponse({'status': 'error', 'message': f'KeyError: {str(e)}'}, status=400)
