from django.contrib.auth import authenticate, login
from django.core.urlresolvers import reverse_lazy
from django.views.generic import FormView

from . import forms


class Register(FormView):
    template_name = 'registration/register.html'
    form_class = forms.RegistrationForm
    success_url = reverse_lazy('home')

    def form_valid(self, form):
        user = form.save()
        user = authenticate(
            username=user.username,
            password=form.cleaned_data['password1'])
        login(self.request, user)
        return super(Register, self).form_valid(form)
