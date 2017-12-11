from celery.decorators import task
from progress.bar import Bar
from sklearn.metrics import jaccard_similarity_score

from network import models


@task
def update_user_based_recommendations():
    experiments = models.Experiment.objects.all().order_by('pk')
    users = models.MyUser.objects.all().order_by('pk')
    user_vectors = dict()

    # Create user vectors
    print('Creating user vectors...')
    bar = Bar('Processing', max=len(users))
    for user in users:
        user_vectors[user] = []
        pk_set = set(models.Experiment.objects.filter(
            experimentfavorite__owner=user).values_list('pk'))
        for exp in experiments:
            if exp.pk in pk_set:
                user_vectors[user].append(1)
            else:
                user_vectors[user].append(0)
        bar.next()
    bar.finish()

    print('Updating user-based recommendations...')
    bar = Bar('Processing', max=len(users))
    for user in users:
        relevant_assemblies = models.Assembly.objects.filter(
            dataset__experiment__owners=user)
        relevant_experiment_types = models.ExperimentType.objects.filter(
            experiment__owners=user)

        relevant_experiments = models.Experiment.objects.filter(
            dataset__assembly__in=relevant_assemblies,
            experiment_type__in=relevant_experiment_types,
        ).exclude(owners=user)

        # Remove any scores that are no longer relevant
        models.UserToExperimentSimilarity.objects.exclude(
            experiment__in=relevant_experiments).delete()

        # Update scores for relevant experiments
        for exp in relevant_experiments:

            favoriting_users = models.MyUser.objects.filter(
                experimentfavorite__favorite=exp)

            sim_score = sum([jaccard_similarity_score(
                user_vectors[user],
                user_vectors[other_user],
            ) for other_user in favoriting_users])

            models.UserToExperimentSimilarity.objects.update_or_create(
                user=user,
                experiment=exp,
                defaults={
                    'score': sim_score,
                }
            )

        bar.next()
    bar.finish()
