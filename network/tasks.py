from celery.decorators import task, periodic_task
from celery import group
from django.utils import timezone
from django.core.cache import cache

from analysis.correlation import Correlation
from analysis import transcript_coverage
from . import models

from datetime import timedelta
from collections import defaultdict
from functools import wraps

import string
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import jaccard_similarity_score
from sklearn.decomposition import PCA

import os
from tempfile import NamedTemporaryFile
import json
from analysis import metaplot
from analysis.normalization import normalize_locus_intersection_values
from analysis.expression import select_transcript_by_expression

from scipy.spatial.distance import mahalanobis
from sklearn.ensemble import RandomForestClassifier

from analysis import score

from subprocess import call

from analysis.encode import Encode

from progress.bar import Bar

ASSAY_REPLACEMENT = {
    'single cell isolation followed by RNA-seq': 'SingleCell RNA-seq',
    'shRNA knockdown followed by RNA-seq': 'shRNA-KD RNA-seq',
    'siRNA knockdown followed by RNA-seq': 'siRNA-KD RNA-seq',
    'CRISPR genome editing followed by RNA-seq': 'CRISPR RNA-seq',
    'whole-genome shotgun bisulfite sequencing': 'WGBS',
    'microRNA-seq': 'miRNA-seq',
}

EXPERIMENT_DESCRIPTION_FIELDS = [
    'assay_slims',
    'assay_synonyms',
    'assay_term_name',
    'assay_title',
    'biosample_summary',
    'biosample_synonyms',
    'biosample_term_name',
    'biosample_type',
    'category_slims',
    'objective_slims',
    'organ_slims',
    'target',
    'system_slims',
]

DATASET_DESCRIPTION_FIELDS = [
    'assembly',
    'biological_replicates',
    'output_category',
    'output_type',
    'technical_replicates',
]


def single_instance_task(cache_id, timeout=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if cache.add(cache_id, True, timeout):
                try:
                    func(*args, **kwargs)
                finally:
                    cache.delete(cache_id)
        return wrapper
    return decorator


@task()
def select_all_representative_transcripts(genes):
    job = group(_select_representative_transcripts.s(g.pk) for g in genes)
    job.apply_async()


@task()
def _select_representative_transcripts(gene_pk):
    select_transcript_by_expression(models.Gene.objects.get(pk=gene_pk))


@task()
def add_or_update_encode():

    def get_encode_url(url):
        return 'https://www.encodeproject.org{}'.format(url)

    project = models.Project.objects.get_or_create(
        name='TEST',
    )[0]

    encode = Encode()
    encode.get_experiments()
    print('{} experiments found in ENCODE!!'.format(len(encode.experiments)))

    experiments_to_process = set()
    datasets_to_process = set()

    for experiment in encode.experiments[:1]:
        encode_id = experiment['name']

        # Create experiment name from fields
        experiment_name = encode_id
        dataset_basename = ''

        # - Add assay to experiment name
        try:
            assay = \
                ASSAY_REPLACEMENT[experiment['detail']['assay_term_name']]
        except KeyError:
            assay = experiment['detail']['assay_term_name']
        assay = assay.replace('-seq', 'seq').replace(' ', '_')
        experiment_name += '-{}'.format(assay)
        dataset_basename += assay

        # - Add target to experiment name
        try:
            target = '-'.join(
                experiment['detail']['target']
                .split('/')[2]
                .split('-')[:-1]
            ).replace('%20', ' ')
        except:
            target = None
        else:
            _target = target.replace(' ', '_').replace('-', '_')
            experiment_name += '-{}'.format(_target)
            dataset_basename += '-{}'.format(_target)

        # - Add cell type or tissue to experiment name
        try:
            biosample_term_name = \
                experiment['detail']['biosample_term_name']
        except:
            biosample_term_name = None
        else:
            _biosample = ('').join(w.replace('-', '').capitalize() for w in
                                   biosample_term_name.split())
            experiment_name += '-{}'.format(_biosample)
            dataset_basename += '-{}'.format(_biosample)

        # Create experiment description from fields
        experiment_description = ''
        for field in EXPERIMENT_DESCRIPTION_FIELDS:
            if field in experiment['detail']:

                if type(experiment['detail'][field]) is list:
                    value = '; '.join(experiment['detail'][field])
                else:
                    value = experiment['detail'][field]

                if field == 'target':
                    value = value.split('/')[2]

                experiment_description += '{}: {}\n'.format(
                    ' '.join(field.split('_')).title(),
                    value.capitalize(),
                )
        experiment_description = experiment_description.rstrip()

        # Get or create associated experiment type object
        try:
            experiment_type = models.ExperimentType.objects.get(
                name=experiment['detail']['assay_term_name'])
        except models.ExperimentType.DoesNotExist:
            experiment_type = models.ExperimentType.objects.create(
                name=experiment['detail']['assay_term_name'],
                short_name=experiment['detail']['assay_term_name'],
                relevant_regions='genebody',
            )

        # Update or create experiment object
        exp, exp_created = models.Experiment.objects.update_or_create(
            project=project,
            slug=experiment['name'],
            defaults={
                'name': experiment_name,
                'project': project,
                'description': experiment_description,
                'experiment_type': experiment_type,
                'cell_type': biosample_term_name,
            },
        )
        if target:
            exp.target = target
            exp.save()
        if exp_created:
            experiments_to_process.add(exp)

        for dataset in experiment['datasets']:

            # Create description for dataset
            dataset_description = ''
            for field in DATASET_DESCRIPTION_FIELDS:
                for detail in ['ambiguous_detail',
                               'plus_detail',
                               'minus_detail']:
                    values = set()
                    try:
                        if type(dataset[detail][field]) is list:
                            values.update(dataset[detail][field])
                        else:
                            values.add(dataset[detail][field])
                        dataset_description += '{}: {}\n'.format(
                            ' '.join(field.split('_')).title(),
                            '\n'.join(
                                str(val) for val in values),
                        )
                    except KeyError:
                        pass
            dataset_description = dataset_description.rstrip()

            # Get associated URLs
            try:
                ambiguous_url = get_encode_url(dataset['ambiguous_href'])
            except:
                ambiguous_url = None
            else:
                slug = dataset['ambiguous']
            try:
                plus_url = get_encode_url(dataset['plus_href'])
                minus_url = get_encode_url(dataset['minus_href'])
            except:
                plus_url = None
                minus_url = None
            else:
                slug = '{}-{}'.format(
                    dataset['plus'],
                    dataset['minus'],
                )

            # Create dataset name
            assembly = dataset['assembly']
            dataset_name = '{}-{}-{}'.format(
                slug,
                dataset_basename,
                assembly,
            )

            # Get assembly object
            try:
                assembly_obj = models.Assembly.objects.get(name=assembly)
            except models.Assembly.DoesNotExist:
                assembly_obj = None
                print(
                    'Assembly "{}" does not exist for dataset {}. '
                    'Skipping dataset.'.format(assembly, dataset_name)
                )

            # Add dataset
            if assembly_obj:

                # Update or create dataset
                ds, ds_created = models.Dataset.objects.update_or_create(
                    slug=slug,
                    defaults={
                        'experiment': exp,
                        'name': dataset_name,
                        'assembly': assembly_obj,
                    },
                )

                # Update URLs, if appropriate
                updated_url = False
                if ambiguous_url:
                    if ds.ambiguous_url != ambiguous_url:
                        ds.ambiguous_url = ambiguous_url
                        updated_url = True
                if plus_url and minus_url:
                    if all([
                        ds.plus_url != plus_url,
                        ds.minus_url != minus_url,
                    ]):
                        ds.plus_url = plus_url
                        ds.minus_url = minus_url
                        updated_url = True
                if updated_url:
                    ds.save()

                # Check if intersections and metaplots already exist
                already_processed = True
                num_total_loci = len(models.Locus.objects.filter(
                    group__assembly=assembly_obj))
                num_existing_intersections = \
                    len(models.DatasetIntersection.objects.filter(
                        dataset=ds))
                if num_total_loci != num_existing_intersections:
                    already_processed = False
                num_total_metaplots = len(models.LocusGroup.objects.filter(
                    assembly=assembly_obj))
                num_existing_metaplots = len(models.MetaPlot.objects.filter(
                    dataset=ds))
                if num_total_metaplots != num_existing_metaplots:
                    already_processed = False

                if ds_created or updated_url or not already_processed:
                    datasets_to_process.add(ds)
                    experiments_to_process.add(exp)

    print('Processing {} datasets...'.format(len(datasets_to_process)))
    process_datasets(list(datasets_to_process))
    print('Processing {} experiments...'.format(len(experiments_to_process)))
    process_experiments(list(experiments_to_process))


@task()
def process_datasets(datasets, chunk=1000):

    assembly_to_intersection_bed = dict()
    for ds in datasets:
        if ds.assembly not in assembly_to_intersection_bed:
            assembly_to_intersection_bed[ds.assembly] = dict()
            for lg in models.LocusGroup.objects.filter(assembly=ds.assembly):
                bed = NamedTemporaryFile(mode='w', delete=False)
                transcript_coverage.generate_locusgroup_bed(lg, bed)
                assembly_to_intersection_bed[ds.assembly][lg] = bed

    assembly_to_metaplot_bed = dict()
    for ds in datasets:
        if ds.assembly not in assembly_to_metaplot_bed:
            assembly_to_metaplot_bed[ds.assembly] = dict()
            for lg in models.LocusGroup.objects.filter(assembly=ds.assembly):
                bed = NamedTemporaryFile(mode='w', delete=False)
                metaplot.generate_metaplot_bed(lg, bed)
                assembly_to_metaplot_bed[ds.assembly][lg] = bed

    bigwig_dir = os.path.join(os.getcwd(), 'data', 'bigwig_temp')
    os.makedirs(bigwig_dir, exist_ok=True)

    for i in range(0, len(datasets), chunk):
        index_1 = i * chunk
        index_2 = min((i + 1) * chunk, len(datasets))
        dataset_chunk = datasets[index_1:index_2]

        download_list_file = NamedTemporaryFile(mode='w')
        bigwig_paths = dict()
        for ds in dataset_chunk:
            if ds.is_stranded():

                download_list_file.write('{}\n'.format(ds.plus_url))
                download_list_file.write('\tdir={}\n'.format(bigwig_dir))
                download_list_file.write('{}\n'.format(ds.minus_url))
                download_list_file.write('\tdir={}\n'.format(bigwig_dir))

                plus_local_path = os.path.join(
                    bigwig_dir, os.path.basename(ds.plus_url))
                minus_local_path = os.path.join(
                    bigwig_dir, os.path.basename(ds.minus_url))

                bigwig_paths[ds.pk] = {
                    'ambiguous': None,
                    'plus': plus_local_path,
                    'minus': minus_local_path,
                }

            else:

                download_list_file.write('{}\n'.format(ds.ambiguous_url))
                download_list_file.write('\tdir={}\n'.format(bigwig_dir))

                ambiguous_local_path = os.path.join(
                    bigwig_dir, os.path.basename(ds.ambiguous_url))

                bigwig_paths[ds.pk] = {
                    'ambiguous': ambiguous_local_path,
                    'plus': None,
                    'minus': None,
                }

        download_list_file.flush()
        call([
            'aria2c',
            '--allow-overwrite=true',
            '--conditional-get=true',
            '-x', '16',
            '-s', '16',
            '-i', download_list_file.name,
        ])
        download_list_file.close()

        tasks = []
        for ds in dataset_chunk:
            intersection_beds = assembly_to_intersection_bed[ds.assembly]
            for lg, bed in intersection_beds.items():
                tasks.append(process_dataset_intersection.s(
                    ds.pk,
                    lg.pk,
                    bed.name,
                    bigwig_paths[ds.pk],
                ))

            metaplot_beds = assembly_to_metaplot_bed[ds.assembly]
            for lg, bed in metaplot_beds.items():
                tasks.append(process_dataset_metaplot.s(
                    ds.pk,
                    lg.pk,
                    bed.name,
                    bigwig_paths[ds.pk],
                ))

        job = group(tasks)
        results = job.apply_async()
        results.join()

        for paths in bigwig_paths.values():
            for field in ['ambiguous', 'plus', 'minus']:
                if paths[field]:
                    os.remove(paths[field])

    for bed_dict in assembly_to_intersection_bed.values():
        for bed in bed_dict.values():
            bed.close()
    for bed_dict in assembly_to_metaplot_bed.values():
        for bed in bed_dict.values():
            bed.close()


@task()
def process_dataset_intersection(dataset_pk, locusgroup_pk, bed_path, bigwigs):
    dataset = models.Dataset.objects.get(pk=dataset_pk)
    loci = models.Locus.objects.filter(group__pk=locusgroup_pk)
    locus_values = transcript_coverage.get_locus_values(
        loci,
        bed_path,
        ambiguous_bigwig=bigwigs['ambiguous'],
        plus_bigwig=bigwigs['plus'],
        minus_bigwig=bigwigs['minus'],
    )
    normalized_values = \
        normalize_locus_intersection_values(loci, locus_values)
    models.DatasetIntersection.objects.filter(
        dataset=dataset, locus__in=loci).delete()
    intersections = []
    for locus in loci:
        intersections.append(models.DatasetIntersection(
            dataset=dataset,
            locus=locus,
            raw_value=locus_values[locus],
            normalized_value=normalized_values[locus],
        ))
    models.DatasetIntersection.objects.bulk_create(intersections)


@task()
def process_dataset_metaplot(dataset_pk, locusgroup_pk, bed_path, bigwigs):
    dataset = models.Dataset.objects.get(pk=dataset_pk)
    locus_group = models.LocusGroup.objects.get(pk=locusgroup_pk)
    metaplot_out = metaplot.get_metaplot_values(
        locus_group,
        bed_path=bed_path,
        ambiguous_bigwig=bigwigs['ambiguous'],
        plus_bigwig=bigwigs['plus'],
        minus_bigwig=bigwigs['minus'],
    )
    models.MetaPlot.objects.update_or_create(
        dataset=dataset,
        locus_group=locus_group,
        defaults={
            'metaplot': json.dumps(metaplot_out),
        },
    )


@task
def process_experiments(experiments):
    set_experiment_intersection_values(experiments)


def get_metadata_similarities():
    EXPERIMENT_DESCRIPTION_FIELDS = [
        'assay_slims',
        'assay_synonyms',
        'assay_term_name',
        'assay_title',
        'biosample_summary',
        'biosample_synonyms',
        'biosample_term_name',
        'biosample_type',
        'category_slims',
        'objective_slims',
        'organ_slims',
        'target',
        'system_slims',
    ]

    descriptions = []
    exps = []

    for exp in models.Experiment.objects.all():
        description = exp.description

        description = description.translate(
            str.maketrans('', '', string.punctuation))
        description = description.split()
        for field in EXPERIMENT_DESCRIPTION_FIELDS:
            description = [x for x in description if x != field]
        if exp.data_type:
            description.append(exp.data_type)
        if exp.cell_type:
            description.append(exp.cell_type)
        if exp.target:
            description.append(exp.target)

        descriptions.append(' '.join(description))
        exps.append(exp)

    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0,
                         stop_words='english')
    tfidf_matrix = tf.fit_transform(descriptions)
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

    similarities = dict()
    for i, row in enumerate(cosine_similarities):
        similarities[exps[i]] = dict()
        for j, value in enumerate(row):
            if i != j:
                similarities[exps[i]][exps[j]] = value
    return similarities


def get_region_variance(gr):
    intersection_values = models.IntersectionValues.objects.filter(
        genomic_regions=gr)

    if intersection_values:

        #  Populate intersection matrix
        intersection_matrix = []
        for iv in intersection_values:
            intersection_matrix.append(iv.intersection_values)

        #  Normalize by sum
        for i, row in enumerate(intersection_matrix):
            _sum = sum(row)
            norm_row = []
            for entry in row:
                norm_row.append(entry / _sum)
            intersection_matrix[i] = norm_row

        #  Calculate variance
        variance = numpy.var(intersection_matrix, 0)

        #  Calculate variance mask
        n = int(0.1 * len(variance))  # Find top 10%; min of 200, max of 1000
        if n < 200:
            n = 200
        elif n > 1000:
            n = 1000
        cutoff = sorted(variance, reverse=True)[n - 1]

        variance_mask = []
        for var in variance:
            if var >= cutoff:
                variance_mask.append(1)
            else:
                variance_mask.append(0)

        #  Update GR model
        gr.variance = list(variance)
        gr.variance_mask = variance_mask
        gr.save()


def get_user_similarites():
    experiments = models.Experiment.objects.all()
    users = models.MyUser.objects.all()
    user_vectors = dict()
    similarities = dict()

    for user in users:
        user_vectors[user] = []
        for exp in experiments:
            if models.ExperimentFavorite.objects.filter(
                    owner=user, favorite=exp).exists():
                user_vectors[user].append(1)
            else:
                user_vectors[user].append(0)

    for user_1 in users:
        similarities[user_1] = dict()
        for user_2 in users:
            similarities[user_1][user_2] = jaccard_similarity_score(
                user_vectors[user_1],
                user_vectors[user_2],
            )

    return similarities


@periodic_task(run_every=timedelta(seconds=5))
@single_instance_task('data_recommendations')
def update_data_recommendations():
    users = models.MyUser.objects.all()
    z_scores = models.ExperimentCorrelation.get_max_z_scores()
    metadata_similarities = get_metadata_similarities()
    user_similarities = get_user_similarites()

    for user in users:
        recs = dict()
        owned_exps = models.Experiment.objects.filter(owners__in=[user])

        #  Find genomic regions where user owns an associated experiment
        user_grs = set()
        for intersection in models.IntersectionValues.objects.filter(
                dataset__experiment__owners__in=[user]):
            user_grs.add(intersection.genomic_regions)

        #  Find all potential experiments for comparison
        compare_exps = set()
        for gr in user_grs:
            for intersection in (models.IntersectionValues.objects
                                 .filter(genomic_regions=gr)
                                 .exclude(dataset__experiment__owners__in=[user])):  # noqa
                compare_exps.add(intersection.dataset.experiment)
        #  Populate rec dict with all potential experiments
        for exp in compare_exps:
            recs[exp] = {
                'correlation_rank': None,
                'correlation_experiment': None,
                'metadata_rank': None,
                'metadata_experiment': None,
                'collaborative_rank': None,
            }

        #  Find dataset correlation rank order
        #  Get z scores for each experiment
        z_score_dict = defaultdict(list)
        for exps, _score in z_scores.items():
            exp_1, exp_2 = exps
            if exp_1 in owned_exps and exp_2 in compare_exps:
                z_score_dict[exp_2].append([_score, exp_1])
            elif exp_1 in compare_exps and exp_2 in owned_exps:
                z_score_dict[exp_1].append([_score, exp_2])

        #  Find max z score for each experiment
        max_z_score_dict = dict()
        for exp, scores in z_score_dict.items():
            _sorted = sorted(scores, key=lambda x: (-x[0], x[1]))
            max_z_score_dict[exp] = {
                'max_z_score': _sorted[0][0],
                'exp': _sorted[0][1],
            }

        #  Get rank from sorted list
        for i, entry in enumerate(sorted(
            max_z_score_dict.items(),
            key=lambda x: (-x[1]['max_z_score'], x[1]['exp'].name))
        ):
            recs[entry[0]]['correlation_rank'] = i
            recs[entry[0]]['correlation_experiment'] = entry[1]['exp']

        #  Find metadata comparison rank order
        #  Get similarities for each experiment
        metadata_sim_dict = defaultdict(list)
        for exp_1 in compare_exps:
            for exp_2 in owned_exps:
                metadata_sim_dict[exp_1].append(
                    (metadata_similarities[exp_1][exp_2], exp_2))

        #  Find max similarities
        max_sim_dict = dict()
        for exp, similarities in metadata_sim_dict.items():
            _sorted = sorted(similarities, key=lambda x: (-x[0], x[1].name))
            max_sim_dict[exp] = {
                'max_sim': _sorted[0][0],
                'exp': _sorted[0][1],
            }

        #  Get rank from sorted list
        for i, entry in enumerate(sorted(
            max_sim_dict.items(),
            key=lambda x: (-x[1]['max_sim'], x[1]['exp'].name))
        ):
            recs[entry[0]]['metadata_rank'] = i
            recs[entry[0]]['metadata_experiment'] = entry[1]['exp']

        #  Find user comparison rank order
        #  Get weighted scores for each experiment
        user_scores = defaultdict(list)
        for exp in compare_exps:
            for comp_user in users:
                if models.ExperimentFavorite.objects.filter().exists():
                    score = 1 * user_similarities[user][comp_user]
                else:
                    score = 0
                user_scores[exp].append(score)

        #  Sum weighted scores for each experiment
        total_user_score = dict()
        for exp, scores in user_scores.items():
            total_user_score[exp] = sum(scores)

        #  Get rank from sorted list
        for i, entry in enumerate(sorted(
            max_sim_dict.items(),
            key=lambda x: -x[1]['max_sim']
        )):
            recs[entry[0]]['collaborative_rank'] = i

        for rec, ranks in recs.items():
            models.ExperimentRecommendation.objects.update_or_create(
                owner=user,
                recommended=rec,
                defaults=ranks,
            )


@periodic_task(run_every=timedelta(seconds=5))
@single_instance_task('user_recommendations')
def update_user_recommendations():
    def add_or_update_user_recommendations():
        # Add a user recommendation if a user's dataset has been favorited
        # Score is the number of favorited dataset_counts
        scores = defaultdict(int)

        for fav in models.DataFavorite.objects.all():
            for dataset_owner in fav.favorite.owners.all():
                if fav.owner != dataset_owner:
                    scores[(fav.owner, dataset_owner)] += 1

        for key, _score in scores.items():
            owner, favorite = key

            ur = models.UserRecommendation.objects.filter(
                owner=owner,
                recommended=favorite,
            )
            uf = models.UserFavorite.objects.filter(
                owner=owner,
                favorite=favorite,
            )

            if not uf.exists():
                if ur.exists():
                    ur[0].score = _score
                    ur[0].save()
                else:
                    models.UserRecommendation.objects.create(
                        owner=owner,
                        recommended=favorite,
                        score=_score,
                    )

    def clean_up_user_recommendations():
        time_threshold = timezone.now() - timedelta(days=7)
        models.UserRecommendation.objects.filter(
            last_updated__lt=time_threshold).delete()

    add_or_update_user_recommendations()
    clean_up_user_recommendations()


@periodic_task(run_every=timedelta(seconds=5))
@single_instance_task('correlation_values')
def update_correlation_values():
    experiments = models.Experiment.objects.all().order_by('id')

    for i, exp_1 in enumerate(experiments):
        intersections_1 = exp_1.get_average_intersections()
        for j, exp_2 in enumerate(experiments[i:]):

            if exp_1 != exp_2:

                intersections_2 = exp_2.get_average_intersections()

                for int_1 in intersections_1:
                    for int_2 in intersections_2:
                        if int_1['regions_pk'] == int_2['regions_pk']:
                            gr = models.GenomicRegions.objects.get(
                                pk=int_1['regions_pk'])
                            if not (models.ExperimentCorrelation.objects
                                    .filter(
                                        x_experiment=exp_1,
                                        y_experiment=exp_2,
                                        genomic_regions=gr,
                                    ).exists()):
                                corr = Correlation(
                                    int_1['intersection_values'],
                                    int_2['intersection_values'],
                                ).get_correlation()[0]
                                models.ExperimentCorrelation.objects.create(
                                    x_experiment=exp_1,
                                    y_experiment=exp_2,
                                    genomic_regions=gr,
                                    score=corr,
                                )


@periodic_task(run_every=timedelta(seconds=5))
@single_instance_task('correlation_values')
def update_metadata_correlation_values():
    experiments = models.Experiment.objects.all().order_by('id')
    similarities = get_metadata_similarities()
    for i, exp_1 in enumerate(experiments):
        for j, exp_2 in enumerate(experiments[i + 1:]):
            models.MetadataCorrelation.objects.update_or_create(
                x_experiment=exp_1,
                y_experiment=exp_2,
                score=similarities[exp_1][exp_2],
            )


@task
def _get_associated_transcript(gene_pk):
    gene = models.Gene.objects.get(pk=gene_pk)
    return gene.get_transcript_with_highest_expression()


@task
def add_or_update_pca(datasets):
    '''
    Perform PCA analysis observing the datasets.
    '''
    # tasks = []
    #
    dataset_pks = set([ds.pk for ds in datasets])

    for lg in sorted(models.LocusGroup.objects.all(), key=lambda x: x.pk):
        for exp_type in sorted(models.ExperimentType.objects.all(),
                               key=lambda x: x.pk):

            subset = set(models.Dataset.objects.filter(
                assembly=lg.assembly,
                experiment__experiment_type=exp_type,
            ).values_list('pk', flat=True))

            # tasks.append(_pca_analysis.s(
            #     lg.pk, exp_type.pk, list(dataset_pks & subset)))

            # print(lg.pk, exp_type.pk)
            #
            _pca_analysis(lg.pk, exp_type.pk, list(dataset_pks & subset))

#     job = group(tasks)
#     results = job.apply_async()
#     results.join()


@task
def _pca_analysis(locusgroup_pk, experimenttype_pk, dataset_pks,
                  size_threshold=200):

    locus_group = models.LocusGroup.objects.get(pk=locusgroup_pk)
    experiment_type = models.ExperimentType.objects.get(pk=experimenttype_pk)

    if locus_group.group_type in ['promoter', 'genebody', 'mRNA']:

        # Get all transcripts associated with the locus group and that are the
        # selected transcript for a gene
        transcripts = models.Transcript.objects.filter(
            gene__annotation__assembly=locus_group.assembly,
            selecting__isnull=False,
        )

        # Filter transcripts by size if not microRNA-seq
        if experiment_type.name != 'microRNA-seq':
            transcripts = [
                t for t in transcripts
                if t.end - t.start + 1 >= size_threshold
            ]

        # Get loci associated with the transcripts and locus group
        loci = models.Locus.objects.filter(
            group=locus_group, transcript__in=transcripts)

    elif locus_group.group_type in ['enhancer']:
        loci = models.Locus.objects.filter(group=locus_group)

    datasets = models.Dataset.objects.filter(pk__in=list(dataset_pks))

    loci_num = len(models.Locus.objects.filter(group=locus_group))
    temp_datasets = []
    for ds in datasets:
        intersection_num = len(models.DatasetIntersection.objects.filter(
            dataset=ds,
            locus__group=locus_group,
        ))
        if loci_num == intersection_num:
            temp_datasets.append(ds)
    datasets = temp_datasets

    if len(datasets) >= 3:
        pca = PCA(n_components=3)
        rf = RandomForestClassifier(n_estimators=1000)

        intersection_values = dict()
        experiment_pks = dict()
        cell_types = dict()
        targets = dict()

        # Get associated data
        for ds in datasets:
            exp = models.Experiment.objects.get(dataset=ds)
            experiment_pks[ds] = exp.pk
            cell_types[ds] = exp.cell_type
            targets[ds] = exp.target

            intersection_values[ds] = dict()

            loci = sorted(loci, key=lambda x: x.pk)
            intersections = (
                models.DatasetIntersection
                      .objects.filter(dataset=ds,
                                      locus__in=loci)
                      .order_by('locus__pk')
            )

            for locus, intersection in zip(loci, intersections):
                intersection_values[ds][locus] = intersection.normalized_value

        # Filter loci by RF importance
        _intersection_values = []
        _cell_types = []
        _targets = []

        for ds in datasets:
            _intersection_values.append([])
            for locus in loci:
                _intersection_values[-1].append(
                    intersection_values[ds][locus])
            _cell_types.append(cell_types[ds])
            _targets.append(targets[ds])

        cell_type_importances = rf.fit(
            _intersection_values, _cell_types).feature_importances_
        target_importances = rf.fit(
            _intersection_values, _cell_types).feature_importances_
        totals = [x + y for x, y in zip(cell_type_importances,
                                        target_importances)]
        filtered_loci = \
            [locus for locus, total in sorted(zip(loci, totals),
                                              key=lambda x: -x[1])][:1000]

        # Filter datasets by Mahalanobis distance after PCA
        filtered_datasets = []
        if len(datasets) >= 10:
            _intersection_values = []
            for ds in datasets:
                _intersection_values.append([])
                for locus in filtered_loci:
                    _intersection_values[-1].append(
                        intersection_values[ds][locus])

            fitted = pca.fit_transform(_intersection_values)

            mean = numpy.mean(fitted, axis=0)
            cov = numpy.cov(fitted, rowvar=False)
            inv = numpy.linalg.inv(cov)
            m_dist = []
            for vector in fitted:
                m_dist.append(mahalanobis(vector, mean, inv))

            Q1 = numpy.percentile(m_dist, 25)
            Q3 = numpy.percentile(m_dist, 75)
            cutoff = Q3 + 1.5 * (Q3 - Q1)

            filtered_datasets = []
            for dist, ds in zip(m_dist, datasets):
                if dist < cutoff:
                    filtered_datasets.append(ds)
        if len(filtered_datasets) <= 1:
            filtered_datasets = datasets

        # Fit PCA with filtered transcripts and filtered datasets
        _intersection_values = []
        for ds in filtered_datasets:
            _intersection_values.append([])
            for locus in filtered_loci:
                _intersection_values[-1].append(
                    intersection_values[ds][locus])
        fitted = pca.fit_transform(_intersection_values)
        if len(fitted) > 1:
            cov = numpy.cov(fitted, rowvar=False)
            if len(_intersection_values) > 3:
                inv = numpy.linalg.inv(cov)
            else:
                inv = numpy.linalg.pinv(cov)
        else:
            cov = None
            inv = None

        # Fit values to PCA, create the associated PCA plot
        fitted_values = []
        pca_plot = []
        for ds in datasets:
            _intersection_values = []
            for locus in filtered_loci:
                _intersection_values.append(
                    intersection_values[ds][locus])
            fitted_values.append(pca.transform([_intersection_values])[0])

            pca_plot.append({
                'dataset_pk': ds.pk,
                'experiment_pk': experiment_pks[ds],
                'experiment_cell_type': cell_types[ds],
                'experiment_target': targets[ds],
                'transformed_values': fitted_values[-1].tolist(),
            })

        # Update or create the PCA object
        pca, created = models.PCA.objects.update_or_create(
            locus_group=locus_group,
            experiment_type=experiment_type,
            defaults={
                'plot': pca_plot,
                'pca': pca,
                'covariation_matrix': cov,
                'inverse_covariation_matrix': inv,
            },
        )

        # Set the PCA to loci relationships
        if not created:
            pca.selected_loci.clear()
        _pca_locus_orders = []
        for i, locus in enumerate(filtered_loci):
            _pca_locus_orders.append(models.PCALocusOrder(
                pca=pca,
                locus=locus,
                order=i,
            ))
        models.PCALocusOrder.objects.bulk_create(
            _pca_locus_orders)


@task
def add_or_update_pca_transformed_values():
    tasks = []

    for pca in models.PCA.objects.all():
        for dataset in models.Dataset.objects.filter(
            assembly=pca.locus_group.assembly,
            experiment__experiment_type=pca.experiment_type,
        ):
            tasks.append(_add_or_update_pca_transformed_values.s(
                dataset.pk, pca.pk
            ))

    job = group(tasks)
    results = job.apply_async()
    results.join()


@task
def _add_or_update_pca_transformed_values(dataset_pk, pca_pk):
    dataset = models.Dataset.objects.get(pk=dataset_pk)
    pca = models.PCA.objects.get(pk=pca_pk)

    order = models.PCALocusOrder.objects.filter(pca=pca).order_by('order')
    loci = [x.locus for x in order]

    intersection_values = []
    missing_values = False
    for locus in loci:
        try:
            intersection_values.append(
                models.DatasetIntersection.objects.get(
                    dataset=dataset, locus=locus).normalized_value
            )
        except models.DatasetIntersection.DoesNotExist:
            print('Missing intersection: Dataset: {}; Locus: {}.'.format(
                str(dataset.pk), str(locus.pk)))
            missing_values = True
            break

    if not missing_values:
        transformed_values = pca.pca.transform([intersection_values])[0]
        models.PCATransformedValues.objects.update_or_create(
            pca=pca,
            dataset=dataset,
            defaults={
                'transformed_values': transformed_values.tolist(),
            },
        )


@task
def update_dataset_data_scores(datasets, quiet=False):
    '''
    Update or create dataset data distance values.
    '''
    updated = set()

    bar_max = 0
    for ds in datasets:
        bar_max += models.Dataset.objects.filter(
            assembly=ds.assembly,
            experiment__experiment_type=ds.experiment.experiment_type,
        ).count()
    bar = Bar('Processing', max=bar_max)

    for ds_1 in datasets:
        for ds_2 in models.Dataset.objects.filter(
            assembly=ds_1.assembly,
            experiment__experiment_type=ds_1.experiment.experiment_type,
        ):

            _ds_1, _ds_2 = sorted([ds_1, ds_2], key=lambda x: x.pk)
            exp_type_1 = _ds_1.experiment.experiment_type
            exp_type_2 = _ds_2.experiment.experiment_type
            if all([
                (_ds_1, _ds_2) not in updated,
                _ds_1 != _ds_2,
                models.PCATransformedValues.objects.filter(
                    dataset=_ds_1,
                    pca__locus_group__group_type=exp_type_1.relevant_regions,
                ).exists(),
                models.PCATransformedValues.objects.filter(
                    dataset=_ds_2,
                    pca__locus_group__group_type=exp_type_2.relevant_regions,
                ).exists(),
            ]):

                distance = score.score_datasets_by_pca_distance(_ds_1, _ds_2)
                models.DatasetDataDistance.objects.update_or_create(
                    dataset_1=_ds_1,
                    dataset_2=_ds_2,
                    defaults={
                        'distance': distance,
                    },
                )
                updated.add((_ds_1, _ds_2))

            bar.next()

    bar.finish()


@task
def update_dataset_metadata_scores(datasets):
    '''
    Update or create dataset metadata distance values.
    '''
    updated = set()

    bar_max = 0
    for ds in datasets:
        bar_max += models.Dataset.objects.filter(
            assembly=ds.assembly,
            experiment__experiment_type=ds.experiment.experiment_type,
        ).count()
    bar = Bar('Processing', max=bar_max)

    onts = []
    onts.append(models.Ontology.objects.get(name='cell_ontology'))
    onts.append(models.Ontology.objects.get(name='cell_line_ontology'))
    onts.append(models.Ontology.objects.get(name='brenda_tissue_ontology'))
    cell_ont_list = [ont.get_ontology_object() for ont in onts]

    gene_ont = models.Ontology.objects.get(
        name='gene_ontology').get_ontology_object()

    for ds_1 in datasets:
        for ds_2 in models.Dataset.objects.filter(
            assembly=ds_1.assembly,
            experiment__experiment_type=ds_1.experiment.experiment_type,
        ):

            _ds_1, _ds_2 = sorted([ds_1, ds_2], key=lambda x: x.pk)
            if all([
                (_ds_1, _ds_2) not in updated,
                _ds_1 != _ds_2,
            ]):

                exp_1 = models.Experiment.objects.get(dataset=_ds_1)
                exp_2 = models.Experiment.objects.get(dataset=_ds_2)

                total_sim = 0

                cell_ont_sims = []
                for ont_obj in cell_ont_list:
                    sim = ont_obj.get_word_similarity(
                        exp_1.cell_type, exp_2.cell_type, metric='lin')
                    if sim:
                        cell_ont_sims.append(sim)
                if cell_ont_sims:
                    total_sim += max(cell_ont_sims)

                gene_ont_sim = gene_ont.get_word_similarity(
                    exp_1.target, exp_2.target, metric='jaccard',
                    weighting='information_content')
                if gene_ont_sim:
                    total_sim += gene_ont_sim

                models.DatasetMetadataDistance.objects.update_or_create(
                    dataset_1=_ds_1,
                    dataset_2=_ds_2,
                    defaults={
                        'distance': total_sim,
                    },
                )

            bar.next()

    bar.finish()


@task
def update_experiment_data_scores(experiments):
    '''
    Update or create experiment data distance values.
    '''
    updated = set()

    bar_max = 0
    for exp in experiments:
        assemblies = models.Assembly.objects.filter(dataset__experiment=exp)
        bar_max += models.Experiment.objects.filter(
            dataset__assembly__in=assemblies,
            experiment_type=exp.experiment_type,
        ).count()
    bar = Bar('Processing', max=bar_max)

    for exp_1 in experiments:
        assemblies = models.Assembly.objects.filter(dataset__experiment=exp_1)
        for exp_2 in models.Experiment.objects.filter(
            dataset__assembly__in=assemblies,
            experiment_type=exp_1.experiment_type,
        ):

            _exp_1, _exp_2 = sorted([exp_1, exp_2], key=lambda x: x.pk)
            rr_1 = _exp_1.experiment_type.relevant_regions
            rr_2 = _exp_2.experiment_type.relevant_regions
            if all([
                (_exp_1, _exp_2) not in updated,
                _exp_1 != _exp_2,
                models.PCATransformedValues.objects.filter(
                    dataset__experiment=_exp_1,
                    pca__locus_group__group_type=rr_1,
                ).exists(),
                models.PCATransformedValues.objects.filter(
                    dataset__experiment=_exp_2,
                    pca__locus_group__group_type=rr_2,
                ).exists(),
            ]):

                distance = score.score_experiments_by_pca_distance(
                    _exp_1, _exp_2)
                models.ExperimentDataDistance.objects.update_or_create(
                    experiment_1=_exp_1,
                    experiment_2=_exp_2,
                    defaults={
                        'distance': distance,
                    },
                )
                updated.add((_exp_1, _exp_2))

            bar.next()

    bar.finish()


@task
def update_experiment_metadata_scores(experiments):
    '''
    Update or create experiment metadata distance values.
    '''
    updated = set()

    bar_max = 0
    for exp in experiments:
        assemblies = models.Assembly.objects.filter(dataset__experiment=exp)
        bar_max += models.Experiment.objects.filter(
            dataset__assembly__in=assemblies,
            experiment_type=exp.experiment_type,
        ).count()
    bar = Bar('Processing', max=bar_max)

    onts = []
    onts.append(models.Ontology.objects.get(name='cell_ontology'))
    onts.append(models.Ontology.objects.get(name='cell_line_ontology'))
    onts.append(models.Ontology.objects.get(name='brenda_tissue_ontology'))
    cell_ont_list = [ont.get_ontology_object() for ont in onts]

    gene_ont = models.Ontology.objects.get(
        name='gene_ontology').get_ontology_object()

    for exp_1 in experiments:
        assemblies = models.Assembly.objects.filter(dataset__experiment=exp_1)
        for exp_2 in models.Experiment.objects.filter(
            dataset__assembly__in=assemblies,
            experiment_type=exp_1.experiment_type,
        ):

            _exp_1, _exp_2 = sorted([exp_1, exp_2], key=lambda x: x.pk)
            if all([
                (_exp_1, _exp_2) not in updated,
                _exp_1 != _exp_2,
            ]):

                total_sim = 0

                cell_ont_sims = []
                for ont_obj in cell_ont_list:
                    sim = ont_obj.get_word_similarity(
                        _exp_1.cell_type, _exp_2.cell_type, metric='lin')
                    if sim:
                        cell_ont_sims.append(sim)
                if cell_ont_sims:
                    total_sim += max(cell_ont_sims)

                gene_ont_sim = gene_ont.get_word_similarity(
                    _exp_1.target, _exp_2.target, metric='jaccard',
                    weighting='information_content')
                if gene_ont_sim:
                    total_sim += gene_ont_sim

                models.ExperimentMetadataDistance.objects.update_or_create(
                    experiment_1=_exp_1,
                    experiment_2=_exp_2,
                    defaults={
                        'distance': total_sim,
                    },
                )
                updated.add((_exp_1, _exp_2))

            bar.next()

    bar.finish()


@task
def set_selected_transcripts_for_genes():
    '''
    For each gene set the selected transcript using expression values.
    '''
    job = group(_set_selected_transcript_for_gene.s(g.pk)
                for g in models.Gene.objects.all())
    job.apply_async()


@task
def _set_selected_transcript_for_gene(gene_pk):
    '''
    Set selected transcript for a single gene.
    '''
    gene = models.Gene.objects.get(pk=gene_pk)
    transcripts = models.Transcript.objects.filter(gene=gene).order_by(
        'name', 'pk')

    if transcripts:
        # If no DatasetIntersection object exists for transcripts, the
        # following will return None
        transcript_w_highest_expression = \
            gene.get_transcript_with_highest_expression()

        if transcript_w_highest_expression:
            transcript = transcript_w_highest_expression
        else:
            transcript = transcripts[0]
    else:
        transcript = None

    gene.selected_transcript = transcript
    gene.save()


@task
def set_experiment_intersection_values(experiments):
    '''
    For each experiment, set average intersection values.
    '''
    job = group(_set_experiment_intersection_values.s(exp.pk)
                for exp in experiments)
    job.apply_async()


@task
def _set_experiment_intersection_values(experiment_pk):
    '''
    For a given experiment, set the average intersection values.
    '''
    exp = models.Experiment.objects.get(pk=experiment_pk)

    # Remove existing
    models.ExperimentIntersection.objects.filter(experiment=exp).delete()

    # Create new
    exp_intersections = []
    for assembly in \
            models.Assembly.objects.filter(dataset__experiment=exp).distinct():
        loci = models.Locus.objects.filter(group__assembly=assembly)
        for locus in loci:
            intersection_values = models.DatasetIntersection.objects.filter(
                locus=locus, dataset__experiment=exp,
            ).values_list('normalized_value', flat=True)
            exp_intersections.append(
                models.ExperimentIntersection(
                    locus=locus,
                    experiment=exp,
                    average_value=numpy.mean(list(intersection_values)),
                )
            )
    models.ExperimentIntersection.objects.bulk_create(exp_intersections)
