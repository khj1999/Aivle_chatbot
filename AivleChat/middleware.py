from django.shortcuts import redirect

class RedirectExceptionMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        return response

    def process_exception(self, request, exception):
        # 모든 예외에 대해 리다이렉트
        return redirect('ChatSystem:home')  # 'ChatSystem' 앱의 'nav' URL로 리다이렉트