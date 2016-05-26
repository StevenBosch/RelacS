from django.views.generic import TemplateView

class IndexView(TemplateView):
    template_name = 'results/index.html'

def index(request):
    return HttpResponse("Hello, world")
