from crispy_forms import layout as cfl
from crispy_forms import bootstrap as cfb
from django.core.urlresolvers import reverse
from django.contrib.auth.forms import AuthenticationForm

from utils.forms import BaseFormHelper


class LoginForm(AuthenticationForm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['password'].widget.attrs['autocomplete'] = 'off'
        self.helper = self.set_helper()

    def set_helper(self):
        buttons = cfb.FormActions(
            cfl.Submit('login', 'Login'),
            cfl.HTML(
                '<a role="button" class="btn btn-default" href="{}">Cancel</a>'
                .format(reverse('home'))),
            cfl.HTML('<br><br>'),
            # cfl.HTML("""
            # <ul>
            #     <li><a href="{0}">Forgot your password?</a></li>
            #     <li><a href="{1}">Create an account</a></li>
            # </ul>
            # """.format(
            #     reverse('user:password_reset'), reverse('user:register'))
            # ),
        )
        helper = BaseFormHelper(
            self,
            horizontal=False,
            legend_text='Login',
            buttons=buttons,
        )
        return helper
