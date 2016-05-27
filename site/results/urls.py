from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.IndexView.as_view(), name='index'),
    url(r'^result/(?P<pk>[0-9]+)$', views.ResultsView.as_view(),
        name='result'),
]
