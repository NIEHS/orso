from django import forms

from . import models
from analysis import metaplot


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
            'data_type', 'cell_type', 'antibody', 'description',
            'ambiguous_url', 'plus_url', 'minus_url', 'name', 'assembly'
        )

    def save(self, commit=True):
        instance = super(DatasetForm, self).save(commit=False)

        assembly = self.cleaned_data.get('assembly')
        annotation = models.GenomeAssembly.objects.get(name=assembly).default_annotation
        promoter_regions = annotation.promoters
        enhancer_regions = annotation.enhancers

        print('Promoter metaplot starting...')
        pm = metaplot.MetaPlot(
            promoter_regions.bed_file.path, single_bw=self.cleaned_data.get('ambiguous_url')
        )
        print('Enhancer metaplot starting...')
        em = metaplot.MetaPlot(
            enhancer_regions.bed_file.path, single_bw=self.cleaned_data.get('ambiguous_url')
        )

        instance.promoter_metaplot = models.MetaPlot.objects.create(
            genomic_regions=promoter_regions,
            bigwig_url=self.cleaned_data.get('ambiguous_url'),
            relative_start=-2500,
            relative_end=2499,
            meta_plot=pm.create_metaplot_json(),
        )
        instance.enhancer_metaplot = models.MetaPlot.objects.create(
            genomic_regions=enhancer_regions,
            bigwig_url=self.cleaned_data.get('ambiguous_url'),
            relative_start=-2500,
            relative_end=2499,
            meta_plot=em.create_metaplot_json(),
        )
        instance.promoter_intersection = models.IntersectionValues.objects.create(
            genomic_regions=promoter_regions,
            bigwig_url=self.cleaned_data.get('ambiguous_url'),
            relative_start=-2500,
            relative_end=2499,
            intersection_values=pm.create_intersection_json(),
        )
        instance.enhancer_intersection = models.IntersectionValues.objects.create(
            genomic_regions=enhancer_regions,
            bigwig_url=self.cleaned_data.get('ambiguous_url'),
            relative_start=-2500,
            relative_end=2499,
            intersection_values=em.create_intersection_json(),
        )
        if commit:
            instance.save()
        return instance
