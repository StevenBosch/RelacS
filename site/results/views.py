from django.views.generic import TemplateView

class IndexView(TemplateView):
    template_name = 'results/index.html'


class ResultsView(TemplateView):
    template_name = 'results/results.html'
