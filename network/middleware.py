from django.core.urlresolvers import resolve
from django.db.models import F

from network import models


class RecordDataAccess(object):

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):

        if request.user.is_authenticated():

            user = models.MyUser.objects.get(user=request.user)
            resolved_path = resolve(request.path)

            try:
                request_pk = int(resolved_path.kwargs['pk'])
            except KeyError:
                request_pk = None
            try:
                request_model = resolved_path.func.view_class.model
            except AttributeError:
                request_model = None

            if request_pk and request_model:

                request_object = request_model.objects.get(pk=request_pk)
                access_object = None

                if request_model == models.Experiment:
                    access_object = \
                        models.ExperimentAccess.objects.get_or_create(
                            user=user,
                            experiment=request_object,
                        )[0]
                elif request_model == models.Dataset:
                    access_object = \
                        models.DatasetAccess.objects.get_or_create(
                            user=user,
                            dataset=request_object
                        )[0]

                if access_object:
                    access_object.access_count = F('access_count') + 1
                    access_object.save(update_fields=['access_count'])

        return self.get_response(request)
