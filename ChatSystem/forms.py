# forms.py
from django import forms
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import UserCreationForm
from .models import *

User = get_user_model()

# Login
# class LoginForm(forms.Form):
#     user_id = forms.CharField(max_length=25, label="User ID")
#     password = forms.CharField(widget=forms.PasswordInput, label="Password")

class LoginForm(forms.Form):
    user_id = forms.CharField(
        max_length=25, 
        label="User ID", 
        widget=forms.TextInput(attrs={'class': 'form-control', 'id': 'id_user_id', 'placeholder': 'Enter your user ID'})
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'id': 'id_password', 'placeholder': 'Enter your password'}),
        label="Password"
    )

# SignUp
class SignUpForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ('user_id', 'email', 'password1', 'password2')

    def save(self, commit=True):
        user = super(SignUpForm, self).save(commit=False)
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
        return user

class UserUpdateForm(forms.ModelForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ['email']

class CsvUploadForm(forms.ModelForm):
    class Meta:
        model = CsvUpload
        fields = ['file']