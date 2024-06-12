# models.py
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager
from django.db import models

class UserManager(BaseUserManager):
    def create_user(self, user_id, email, password=None):
        if not user_id:
            raise ValueError('The User ID must be set')
        if not email:
            raise ValueError('The Email must be set')

        user = self.model(
            user_id=user_id,
            email=self.normalize_email(email),
        )
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, user_id, email, password=None):
        user = self.create_user(
            user_id=user_id,
            email=email,
            password=password,
        )
        user.is_admin = True
        user.save(using=self._db)
        return user

# 사용자
class User(AbstractBaseUser):
    user_id = models.CharField(max_length=25, unique=True)
    email = models.EmailField(unique=True)
    is_active = models.BooleanField(default=True)
    is_admin = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    objects = UserManager()

    USERNAME_FIELD = 'user_id'
    REQUIRED_FIELDS = ['email']

    def __str__(self):
        return self.user_id

    def has_perm(self, perm, obj=None):
        return self.is_admin

    def has_module_perms(self, app_label):
        return True

    @property
    def is_staff(self):
        return self.is_admin

    
# 채팅
class Chat(models.Model):
    user = models.ForeignKey(User, related_name='chats', on_delete=models.CASCADE)
    chat_name = models.CharField(max_length=10)
    chat_id = models.CharField(max_length=255, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.chat_name
    

# 채팅 메시지
class Message(models.Model):
    chat = models.ForeignKey(Chat, related_name='messages', on_delete=models.CASCADE)
    sender = models.CharField(max_length=255)  # 'human' 또는 'ai' 저장
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.sender}: {self.content[:50]}...'  # 메시지 내용을 일부 포함
    
class AivleChatHistory(models.Model):
    datetime = models.DateTimeField(auto_now_add=True)  # 자동으로 현재 시간 설정
    query = models.TextField()
    sim1 = models.FloatField()
    sim2 = models.FloatField()
    sim3 = models.FloatField()
    answer = models.TextField()

    def __str__(self):
        return (f'History(id={self.id}, datetime={self.datetime}, query={self.query}, '
                f'sim1={self.sim1}, sim2={self.sim2}, sim3={self.sim3}, answer={self.answer})')
    
class CsvUpload(models.Model):
    file = models.FileField(upload_to='csvs/')
    uploaded_at = models.DateTimeField(auto_now_add=True)