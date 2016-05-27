from django.views.generic import TemplateView, ListView, DetailView

from results import models

class IndexView(ListView):
    template_name = 'results/index.html'
    model = models.Recording
    context_object_name = 'recordings'


class ResultsView(DetailView):
    template_name = 'results/results.html'
    model = models.Recording
    context_object_name = 'recording'

    def get_context_data(self, **kwargs):
        context = super(ResultsView, self).get_context_data(**kwargs)
        print self.object
        context['representation'] = self.object.filename + '.png'
        return context
