from crispy_forms import layout as cfl
from crispy_forms import bootstrap as cfb
from django import forms
from django.core.urlresolvers import reverse
from django.contrib.auth.forms import AuthenticationForm, User

from network import models
from utils.forms import BaseFormHelper

PASSWORD_HELP = 'Must have at least 8 characters'
PUBLIC_HELP = 'Public accounts may be followed by other users'


def check_password(pw):
    """Ensure password meets complexity requirements."""
    if len(pw) < 8:
        return False
    else:
        return True


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


class RegistrationForm(forms.ModelForm):

    email = forms.CharField(
        label='Email address',
        required=True)
    password1 = forms.CharField(
        label='Password',
        widget=forms.PasswordInput(attrs={'autocomplete': 'off'}),
        help_text=PASSWORD_HELP)
    password2 = forms.CharField(
        label='Password confirmation',
        widget=forms.PasswordInput(attrs={'autocomplete': 'off'}))
    public = forms.BooleanField(
        label='Public account?',
        initial=True,
        help_text=PUBLIC_HELP,
        required=False)

    class Meta:
        model = User
        fields = ('username', 'email')

    def __init__(self, *args, **kwargs):
        super(RegistrationForm, self).__init__(*args, **kwargs)
        self.helper = self.set_helper()

    def set_helper(self):
        buttons = cfb.FormActions(
            cfl.Submit('login', 'Create account'),
            cfl.HTML(
                '<a role="button" class="btn btn-default" href="{}">Cancel</a>'
                .format(reverse('login'))),
        )

        helper = BaseFormHelper(
            self,
            horizontal=False,
            legend_text='Create an account',
            buttons=buttons)
        helper.form_class = 'loginForm'

        return helper

    def clean(self):
        password1 = self.cleaned_data.get('password1')
        password2 = self.cleaned_data.get('password2')
        if password1 and password2 and password1 != password2:
            msg = 'Passwords do not match'
            self.add_error('password2', msg)

        if not check_password(password1):
            msg = 'Password is not long enough'
            self.add_error('password1', msg)

        username = self.cleaned_data.get('username')
        if User.objects.filter(username__iexact=username).count() > 0:
            msg = 'Username already exists'
            self.add_error('username', msg)

    def save(self, commit=True):
        user = super(RegistrationForm, self).save(commit=False)
        user.set_password(self.cleaned_data['password1'])
        public = self.cleaned_data['public']
        if commit:
            user.save()
            models.MyUser.objects.create(
                user=user,
                slug=user.username,
                public=public,
            )
        return user
