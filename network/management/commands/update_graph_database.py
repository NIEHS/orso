from django.core.management.base import BaseCommand
from neomodel import UniqueProperty, Traversal
from neomodel.match import OUTGOING

from network import models


def create_users():

    for my_user in models.MyUser.objects.all():
        try:
            models.GraphUser(
                rel_id=my_user.pk,
                username=my_user.user.username,
            ).save()
        except UniqueProperty:
            pass


def create_experiments():

    for exp in models.Experiment.objects.all():
        try:
            exp_node = models.GraphExperiment(
                rel_id=exp.pk,
                name=exp.name,
                experiment_type_id=exp.experiment_type.pk,
                organism_id=exp.organism.pk,
            ).save()
        except UniqueProperty:
            exp_node = models.GraphExperiment.nodes.get(
                rel_id=exp.pk)

        for owner in exp.owners.all():
            user_node = models.GraphUser.nodes.get(rel_id=owner.pk)
            user_node.owned_experiment.connect(exp_node)

        for ds in models.Dataset.objects.filter(experiment=exp):

            try:
                ds_node = models.GraphDataset(
                    rel_id=ds.pk,
                    name=ds.name,
                    assembly_id=ds.assembly.pk,
                ).save()
            except UniqueProperty:
                ds_node = models.GraphDataset.nodes.get(
                    rel_id=ds.pk)

            ds_node.experiment.connect(exp_node)


def clear_relationships(from_node_model, to_node_model, relation_type,
                        relation_attr):

    for node_1 in from_node_model.nodes.all():
        traversal = Traversal(
            node_1,
            from_node_model.__label__,
            {
                'node_class': to_node_model,
                'direction': OUTGOING,
                'relation_type': relation_type,
            }
        )
        for node_2 in traversal.all():
            getattr(node_1, relation_attr).disconnect(node_2)


def update_favorites():

    clear_relationships(models.GraphUser, models.GraphExperiment,
                        'FAVORITE', 'favorited_experiment')

    for fav in models.Favorite.objects.all():
        user_node = models.GraphUser.nodes.get(rel_id=fav.user.pk)
        exp_node = models.GraphExperiment.nodes.get(rel_id=fav.experiment.pk)

        if not user_node.favorited_experiment.is_connected(exp_node):
            user_node.favorited_experiment.connect(exp_node)


def update_follows():

    clear_relationships(models.GraphUser, models.GraphUser,
                        'FOLLOW', 'followed_user')

    for follow in models.Follow.objects.all():
        from_user_node = models.GraphUser.nodes.get(rel_id=follow.following.pk)
        to_user_node = models.GraphUser.nodes.get(rel_id=follow.followed.pk)

        if not from_user_node.followed_user.is_connected(to_user_node):
            from_user_node.followed_user.connect(to_user_node)


def update_similarities():

    clear_relationships(models.GraphExperiment, models.GraphExperiment,
                        'SIMILARITY', 'sim_to_experiment')

    for sim in models.Similarity.objects.filter(sim_type='metadata'):
        exp_1 = sim.experiment_1
        exp_2 = sim.experiment_2

        exp_node_1 = models.GraphExperiment.nodes.get(rel_id=exp_1.pk)
        exp_node_2 = models.GraphExperiment.nodes.get(rel_id=exp_2.pk)

        exp_node_1.sim_to_experiment.connect(exp_node_2, {'score': 1})
        exp_node_2.sim_to_experiment.connect(exp_node_1, {'score': 1})

    clear_relationships(models.GraphDataset, models.GraphDataset,
                        'SIMILARITY', 'sim_to_dataset')

    for sim in models.Similarity.objects.filter(sim_type='primary'):
        ds_1 = sim.dataset_1
        ds_2 = sim.dataset_2

        ds_node_1 = models.GraphDataset.nodes.get(rel_id=ds_1.pk)
        ds_node_2 = models.GraphDataset.nodes.get(rel_id=ds_2.pk)

        ds_node_1.sim_to_dataset.connect(ds_node_2, {'score': 1})
        ds_node_2.sim_to_dataset.connect(ds_node_1, {'score': 1})


def update_recommendations():

    clear_relationships(models.GraphExperiment, models.GraphUser,
                        'RECOMMENDATION', 'recommended_user')

    for rec in models.Recommendation.objects.all():
        exp_node = models.GraphExperiment.nodes.get(
            rel_id=rec.recommended_experiment.pk)
        user_node = models.GraphUser.nodes.get(rel_id=rec.user.pk)

        if not exp_node.recommended_user.is_connected(user_node):
            exp_node.recommended_user.connect(user_node)


class Command(BaseCommand):
    help = '''
        Use objects in the relational DB to construct the graph DB.
    '''

    def handle(self, *args, **options):

        create_users()
        create_experiments()
        update_favorites()
        update_follows()
        update_similarities()
        update_recommendations()
