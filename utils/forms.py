
from crispy_forms import helper as cf
from crispy_forms import layout as cfl
from crispy_forms import bootstrap as cfb


class BaseFormHelper(cf.FormHelper):

    error_text_inline = True

    def __init__(self, form, horizontal=True, **kwargs):
        self.attrs = {}
        self.inputs = []
        self.kwargs = kwargs

        # horizontal boostrap form
        if horizontal:
            self.form_class = 'form-horizontal'
            self.label_class = 'col-lg-2'
            self.field_class = 'col-lg-10'

        self.form = form
        self.layout = self.build_default_layout()

        buttons = self.kwargs.get('buttons')
        if buttons is None:
            buttons = self.get_buttons()
        self.layout.append(buttons)

    def build_default_layout(self):
        layout = cfl.Layout(*self.form.fields.keys())

        if self.kwargs.get('legend_text'):
            layout.insert(0, cfl.HTML(u"<legend>{}</legend>".format(
                self.kwargs.get('legend_text'))))

        if self.kwargs.get('help_text'):
            layout.insert(1, cfl.HTML("""
                <p class="help-block">{}</p><br>
            """.format(self.kwargs.get('help_text'))))

        return layout

    def get_buttons(self):
        btns = []

        # save button
        btns.append(cfl.Submit('save', 'Save'))

        # cancel button
        url = self.form.instance.get_form_cancel_url()
        btns.append(
            cfl.HTML("""
                <a role="button" class="btn btn-default" href="{}">Cancel</a>
            """.format(url)))

        return cfb.FormActions(*btns, css_class="form-actions")
