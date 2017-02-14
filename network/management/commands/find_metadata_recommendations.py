from django.core.management.base import BaseCommand
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from network import models
import re
from .add_project import EXPERIMENT_DESCRIPTION_FIELDS


class Command(BaseCommand):
    def handle(self, *args, **options):
        descriptions = []
        pks = []

        for exp in models.Experiment.objects.all():
            description = exp.description
            for field in EXPERIMENT_DESCRIPTION_FIELDS:
                description = description.replace(field + ':', '')
            description = description.replace('\n', ' ').strip()
            description = re.sub(' +', ' ', description)

            descriptions.append(description)
            pks.append(exp.pk)

        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0,
                             stop_words='english')
        tfidf_matrix = tf.fit_transform(descriptions)
        cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

        similarities = dict()
        for i, row in enumerate(cosine_similarities[:1]):
            similarities[pks[i]] = []
            for j, value in enumerate(row):
                if i != j:
                    similarities[pks[i]].append({
                        'id': pks[j],
                        'score': value,
                    })

        pk, scores = list(similarities.items())[0]
        sorted_scores = sorted(scores, key=lambda k: k['score'], reverse=True)

        exp_1 = models.Experiment.objects.get(pk=pk)
        exp_2 = models.Experiment.objects.get(pk=sorted_scores[1]['id'])

        print(exp_1.description)
        print(exp_2.description)
