from django.urls import path
from . import views

app_name = 'ChatSystem'

urlpatterns = [
    # 홈페이지
    path('', views.home, name='home'),
    path('nav', views.nav, name='nav'),

    # 회원가입, 로그인
    path('user_info', views.user_info, name='user_info'),
    path('signup', views.signup, name='signup'),
    path('login', views.user_login, name='login'),
    path('logout', views.user_logout, name='logout'),
    path('delete_user', views.delete_user, name='delete_user'),

    # 챗봇
    path('chat', views.index_chat, name='chat'),
    path('chat_ajax', views.chat_ajax, name='chat_ajax'),

    # aivle QnA
    path('aivle_chat', views.index_aivle_chat, name='aivle_chat'),
    path('aivle_chat_ajax', views.aivle_chat_ajax, name='aivle_chat_ajax'),

    # 채팅 관리
    path('create_chat', views.create_chat, name='create_chat'),
    path('delete_chat/<str:chat_id>/', views.delete_chat, name='delete_chat'),
    path('load_chat_history', views.load_chat_history, name='load_chat_history'),
    path('load_chat_list', views.load_chat_list, name='load_chat_list'),

]
