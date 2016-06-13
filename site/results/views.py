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
        context['representation'] = self.object.filename + '.png'
        sounds = []
        for s in self.object.sound_set.all():
            left = s.start / float(self.object.length) * 100
            width = s.end / float(self.object.length) * 100 - left
            r = int(s.stressful * 2.55)
            g = 255 - r
            b = 0
            sound = {'left': left,
                'width': width,
                'stressful': s.stressful,
                'relaxing': s.relaxing,
                'categories': s.category_set,
                'str': str(s),
                'color': 'rgb({},{},{})'.format(r, g, b),
            }
            sounds.append(sound)
        context['sounds'] = sounds
        return context
