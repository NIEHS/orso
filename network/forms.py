from django import forms

from . import models
# from analysis import metaplot


class DatasetForm(forms.ModelForm):

    URL_HELP = 'URL for downloading user-dataset, must be publicly available without authentication.'  # noqa

    ambiguous_url = forms.URLField(
        required=False,
        label='URL (strands unspecified)',
        help_text=URL_HELP)
    plus_url = forms.URLField(
        required=False,
        label='URL (plus-strand)',
        help_text=URL_HELP)
    minus_url = forms.URLField(
        required=False,
        label='URL (minus-strand)',
        help_text=URL_HELP)

    class Meta:
        model = models.Dataset
        fields = (
            'data_type', 'cell_type', 'target', 'description',
            'ambiguous_url', 'plus_url', 'minus_url', 'name', 'assembly'
        )
