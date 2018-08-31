import dateutil.parser
import requests

from django.db.models import Q

from network import models
from network.tasks.processing import process_dataset_batch

HEADERS = {'accept': 'application/json'}
SELECTION_HIERARCHY = [
    ['plus strand signal of unique reads',
        'minus strand signal of unique reads'],
    ['plus strand signal of all reads', 'minus strand signal of all reads'],
    ['plus strand signal', 'minus strand signal'],
    ['fold change over control'],
    ['signal p-value'],
    ['control normalized signal'],
    ['read-depth normalized signal'],
    ['percentage normalized signal'],
    ['signal of unique reads'],
    ['signal of all reads'],
    ['signal'],
    ['base overlap signal'],
    ['genome compartments'],
    ['wavelet-smoothed signal'],
    ['base overlap signal'],
    ['raw signal']
]
ASSEMBLY_ALIASES = {
    'mm10-minimal': 'mm10',
}

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


def get_selection_level(output_type):
    for level, group in enumerate(SELECTION_HIERARCHY):
        if output_type in group:
            return level


def get_encode_url(href):
    return 'https://www.encodeproject.org{}'.format(href)


class EncodeExperiment(object):

    def __init__(self, entry):
        self.datasets = []
        self.controls = []

        self.experiment_type = entry['assay_term_name']
        self.id = entry['accession']

        self._set_cell_type(entry)
        self._set_controls(entry)
        self._set_description(entry)
        self._set_experiment_type(entry)
        self._set_files(entry)
        self._set_target(entry)

        self._set_name()

    def _set_cell_type(self, entry):
        if 'biosample_term_name' in entry:
            self.cell_type = entry['biosample_term_name']
            self.short_cell_type = \
                ('').join(w.replace('-', '').capitalize()
                          for w in entry['biosample_term_name'].split())
        else:
            self.cell_type = None
            self.short_cell_type = None

    def _set_controls(self, entry):
        self.controls = []
        for control in entry['possible_controls']:
            self.controls.append(control.split('/')[2])

    def _set_description(self, entry):
        self.description = ''
        for field in EXPERIMENT_DESCRIPTION_FIELDS:
            if field in entry:
                if type(entry[field]) is list:
                    value = '; '.join(entry[field])
                else:
                    value = entry[field]
                if field == 'target':
                    value = value.split('/')[2]
                self.description += '{}: {}\n'.format(
                    ' '.join(field.split('_')).title(),
                    value.capitalize(),
                )
        self.description.rstrip()

    def _set_target(self, entry):
        if 'target' in entry:
            self.target = '-'.join(
                entry['target']
                .split('/')[2]
                .split('-')[:-1]
            ).replace('%20', ' ')
        else:
            self.target = None

    def _set_files(self, entry):
        self.files = []
        for _file in entry['files']:
            self.files.append(_file.split('/')[2])

    def _set_experiment_type(self, entry):
        self.experiment_type = entry['assay_term_name']

        term = self.experiment_type
        if term in ASSAY_REPLACEMENT:
            term = ASSAY_REPLACEMENT[term]
        self.short_experiment_type = \
            term.replace('-seq', 'seq').replace(' ', '_')

    def _set_name(self):
        fields = [
            self.id,
            self.short_experiment_type,
            self.target,
            self.short_cell_type,
        ]
        fields = [x for x in fields if x is not None]
        self.name = '-'.join(fields)

    def set_organism(self):
        potential_organisms = set()
        for ds in self.datasets:
            if ds.assembly in ['hg19', 'GRCh38']:
                potential_organisms.add('Homo sapiens')
            elif ds.assembly in ['mm9', 'mm10']:
                potential_organisms.add('Mus musculus')
            elif ds.assembly in ['dm3', 'dm6']:
                potential_organisms.add('Drosophila melanogaster')
            elif ds.assembly in ['ce10', 'ce11']:
                potential_organisms.add('Caenorhabditis elegans')
        if len(potential_organisms) > 1:
            raise ValueError('More than one possible organism.')
        elif len(potential_organisms) == 0:
            raise ValueError('No assembly associated with experiment.')
        self.organism = list(potential_organisms)[0]


class EncodeDataset(object):

    def __init__(self, **kwargs):

        self.assembly = None
        self.cell_type = None
        self.experiment_type = None
        self.replicate = None
        self.target = None

        self.plus = None
        self.minus = None
        self.ambiguous = None

        for key in [
            'assembly',
            'replicate',
            'target',

            'short_cell_type',
            'short_experiment_type',

            'plus',
            'minus',
            'ambiguous',
        ]:
            if key in kwargs:
                setattr(self, key, kwargs[key])

        self._set_id()
        self._set_name()
        self._set_description()

    def _set_id(self):
        if self.plus and self.minus:
            self.id = '-'.join([self.plus.id, self.minus.id])
        elif self.ambiguous:
            self.id = self.ambiguous.id

    def _set_name(self):
        fields = [
            self.id,
            self.short_experiment_type,
            self.target,
            self.short_cell_type,
        ]
        fields = [x for x in fields if x is not None]
        self.name = '-'.join(fields)

    def _set_description(self):
        self.description = ''

        if self.plus and self.minus:
            _bigwig = self.plus
        else:
            _bigwig = self.ambiguous

        for field in DATASET_DESCRIPTION_FIELDS:

            if getattr(self, field, None):
                value = getattr(self, field)
            elif getattr(_bigwig, field, None):
                value = getattr(_bigwig, field)
            else:
                value = None
            if value:
                if type(value) is list or type(value) is tuple:
                    value = ', '.join([str(x) for x in value])
                self.description += '{}: {}\n'.format(
                    ' '.join(field.split('_')).title(),
                    value.capitalize(),
                )

        self.description.rstrip()


class EncodeFile(object):

    def __init__(self, entry):
        self.id = entry['accession']
        self.file_size = int(entry['file_size'])
        self.href = entry['href']
        self.output_category = entry['output_category']
        self.output_type = entry['output_type']

        self._set_assembly(entry)
        self._set_date_created(entry)
        self._set_replicate(entry)
        self._set_strand(entry)

        self._set_selection_level()
        self._set_url()

    def _set_assembly(self, entry):
        self.assembly = entry['assembly']
        if self.assembly in ASSEMBLY_ALIASES:
            self.assembly = ASSEMBLY_ALIASES[self.assembly]

    def _set_date_created(self, entry):
        self.date_created = \
            dateutil.parser.parse(entry['date_created']).replace(tzinfo=None)

    def _set_replicate(self, entry):
        self.biological_replicates = tuple(entry['biological_replicates'])
        self.technical_replicates = tuple(entry['technical_replicates'])

    def _set_strand(self, entry):
        if 'plus' in self.output_type:
            self.strand = 'plus'
        elif 'minus' in self.output_type:
            self.strand = 'minus'
        else:
            self.strand = 'ambiguous'

    def _set_selection_level(self):
        self.selection_level = float('Inf')
        for level, group in enumerate(SELECTION_HIERARCHY):
            if self.output_type in group:
                self.selection_level = level
                break

    def _set_url(self):
        self.url = get_encode_url(self.href)


class EncodeProject(object):

    def __init__(self):
        self._bigwigs = dict()
        self.experiments = []

        self._call_api()
        self._clean_up()

    def _call_api(self):
        self._retrieve_experiments()
        self._retrieve_bigwigs()
        self._create_datasets()

    def _clean_up(self):
        self._filter_datasets_by_missing_strand_data()
        self._filter_experiments_without_datasets()
        self._set_empty_replicate_values()
        self._set_experiment_organisms()

    def _retrieve_experiments(self):
        URL = (
            'https://www.encodeproject.org/search/?'
            'type=experiment&'
            'format=json&'
            'limit=all&'
            'frame=object'
        )
        response = requests.get(URL, headers=HEADERS).json()
        for entry in response['@graph']:
            self.experiments.append(EncodeExperiment(entry))

    def _retrieve_bigwigs(self):
        URL = (
            'https://www.encodeproject.org/search/?'
            'type=file&'
            'format=json&'
            'limit=all&'
            'frame=object&'
            'file_format=bigWig'
        )
        response = requests.get(URL, headers=HEADERS).json()
        for entry in response['@graph']:
            key = entry['accession']
            self._bigwigs.update({key: EncodeFile(entry)})

    def _create_datasets(self):
        for experiment in self.experiments:
            bigwigs = [self._bigwigs[x] for x in experiment.files
                       if x in self._bigwigs]
            replicates = dict()

            # Each bigWig is given a replicate ID. Associate all bigWigs for a
            # replicate, observing assembly and strand.
            for bigwig in bigwigs:
                key = (bigwig.biological_replicates, bigwig.assembly)
                if key not in replicates:
                    replicates[key] = {
                        'plus': [],
                        'minus': [],
                        'ambiguous': [],
                    }
                replicates[key][bigwig.strand].append(bigwig)

            # Only one file per replicate-strand-assembly should be considered.
            # Selection criteria include output type and date created.
            for key, stranded_files in replicates.items():
                for strand, entries in stranded_files.items():
                    try:
                        replicates[key][strand] = \
                            sorted(entries, key=lambda x: (
                                -x.selection_level,
                                x.date_created,
                                x.file_size,
                                x.id,
                            ))[-1]
                    except IndexError:
                        pass

            # If complete stranded information is available,
            # ignore non-stranded.
            for key, stranded_files in replicates.items():
                if stranded_files['plus'] and stranded_files['minus']:
                    stranded_files['ambiguous'] = []

            # Some 'replicates' are formed by merging others. '(1, 2)' would
            # mean that the replicate is the product of merging 1 and 2. Keep
            # merged replicates and remove their components.
            remove_entries = set()
            for replicate, assembly in replicates.keys():
                for other_replicate, other_assembly in replicates.keys():
                    remove = True
                    for entry in replicate:
                        if entry not in other_replicate:
                            remove = False
                    if replicate != other_replicate and \
                            assembly == other_assembly and remove:
                        remove_entries.add((replicate, assembly))
            for key in remove_entries:
                del(replicates[key])

            for key, strand_files in replicates.items():
                replicate, assembly = key
                experiment.datasets.append(EncodeDataset(
                    biological_replicates=replicate,
                    assembly=assembly,
                    short_experiment_type=experiment.short_experiment_type,
                    target=experiment.target,
                    short_cell_type=experiment.short_cell_type,
                    plus=strand_files['plus'],
                    minus=strand_files['minus'],
                    ambiguous=strand_files['ambiguous'],
                ))

    def _filter_datasets_by_missing_strand_data(self):
        for experiment in self.experiments:
            kept_datasets = []
            for dataset in experiment.datasets:
                if any([
                    (dataset.plus and dataset.minus and
                        not dataset.ambiguous),
                    (dataset.ambiguous and
                        not dataset.plus and not dataset.minus),
                ]):
                    kept_datasets.append(dataset)
            experiment.datasets = kept_datasets

    def _filter_experiments_without_datasets(self):
        kept_experiments = []
        for experiment in self.experiments:
            if experiment.datasets:
                kept_experiments.append(experiment)
        self.experiments = kept_experiments

    def _set_empty_replicate_values(self):
        for experiment in self.experiments:
            for dataset in experiment.datasets:
                if dataset.replicate is None:
                    dataset.replicate = (1,)

    def _set_experiment_organisms(self):
        for experiment in self.experiments:
            experiment.set_organism()


def add_or_update_encode():

    project = models.Project.objects.get_or_create(
        name='ENCODE',
    )[0]

    encode = EncodeProject()
    print('{} experiments found in ENCODE!!'.format(len(encode.experiments)))

    experiments_to_process = set()
    datasets_to_process = set()

    # Create experiment and dataset objects; process datasets
    for experiment in encode.experiments[:100]:

        organism_obj = models.Organism.objects.get(name=experiment.organism)

        try:
            experiment_type_obj = models.ExperimentType.objects.get(
                name=experiment.experiment_type)
        except models.ExperimentType.DoesNotExist:
            experiment_type_obj = models.ExperimentType.objects.create(
                name=experiment.experiment_type,
                short_name=experiment.short_experiment_type,
                relevant_regions='genebody',
            )

        # Update or create experiment object
        exp_obj, exp_created = models.Experiment.objects.update_or_create(
            project=project,
            consortial_id=experiment.id,
            defaults={
                'name': experiment.name,
                'organism': organism_obj,
                'project': project,
                'description': experiment.description,
                'experiment_type': experiment_type_obj,
                'cell_type': experiment.cell_type,
                'slug': experiment.name,
            },
        )
        if experiment.target:
            exp_obj.target = experiment.target
            exp_obj.save()

        for dataset in experiment.datasets:

            # Get assembly object
            try:
                assembly_obj = \
                    models.Assembly.objects.get(name=dataset.assembly)
            except models.Assembly.DoesNotExist:
                assembly_obj = None
                print(
                    'Assembly "{}" does not exist for dataset {}. '
                    'Skipping dataset.'.format(dataset.assembly, dataset.name)
                )

            # Add dataset
            if assembly_obj:

                # Update or create dataset
                ds_obj, ds_created = models.Dataset.objects.update_or_create(
                    consortial_id=dataset.id,
                    experiment=exp_obj,
                    defaults={
                        'name': dataset.name,
                        'assembly': assembly_obj,
                        'slug': dataset.id,
                    },
                )

                # Update URLs, if appropriate
                updated_url = False
                if dataset.ambiguous:
                    if ds_obj.ambiguous_url != dataset.ambiguous.url:
                        ds_obj.ambiguous_url = dataset.ambiguous.url
                        updated_url = True
                elif dataset.plus and dataset.minus:
                    if any([
                        ds_obj.plus_url != dataset.plus.url,
                        ds_obj.minus_url != dataset.minus.url,
                    ]):
                        ds_obj.plus_url = dataset.plus.url
                        ds_obj.minus_url = dataset.minus.url
                        updated_url = True
                if updated_url:
                    ds_obj.processed = False
                    ds_obj.save()

                if not ds_obj.processed:
                    datasets_to_process.add(ds_obj)

    print('Processing {} datasets...'.format(len(datasets_to_process)))
    process_dataset_batch(list(datasets_to_process))

    revoke_missing_experiments(encode, project)
    revoke_missing_datasets(encode, project)
    revoke_experiments_with_revoked_datasets(encode, project)

    # Set 'processed' flag for experiments
    for exp_obj in models.Experiment.objects.filter(project=project):
        ds_objs = models.Dataset.objects.filter(experiment=exp_obj)
        exp_obj.processed = all([ds_obj.processed for ds_obj in ds_objs])
        exp_obj.save()


def revoke_missing_experiments(encode_obj, project):
    query = Q()
    for experiment in encode_obj.experiments:
        query |= Q(consortial_id=experiment.id)
    query_set = (models.Experiment.objects
                                  .filter(project=project)
                                  .exclude(query))
    for experiment in query_set:
        experiment.revoked = True
        experiment.save()
    print('{} experiments missing from query. Revoked!!'.format(
        str(query_set.count())))


def revoke_missing_datasets(encode_obj, project):
    query = Q()
    for experiment in encode_obj.experiments:
        for dataset in experiment.datasets:
            for attr in ['ambiguous', 'plus', 'minus']:
                try:
                    query |= Q(
                        consortial_id__contains=getattr(dataset, attr).id)
                except AttributeError:
                    pass
    query_set = (models.Dataset.objects
                               .filter(experiment__project=project)
                               .exclude(query))
    for dataset in query_set:
        dataset.revoked = True
        dataset.save()
    print('{} datasets missing from query. Revoked!!'.format(
        str(query_set.count())))


def revoke_experiments_with_revoked_datasets(encode_obj, project):
    query_set = (models.Experiment.objects
                                  .filter(revoked=False, project=project)
                                  .exclude(dataset__revoked=False))
    for experiment in query_set:
        experiment.revoked = True
        experiment.save()
    print('{} experiments with only revoked datasets. Revoked!!'.format(
        str(query_set.count())))
