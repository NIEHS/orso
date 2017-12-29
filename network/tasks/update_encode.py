import dateutil.parser
import json
import requests

from celery.decorators import task

from network import models
from network.tasks.process_datasets import process_datasets

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


class Encode(object):

    def __init__(self):
        self.exp_dict = dict()
        self.bw_dict = dict()
        self.experiments = []

    def call_api(self):
        bw_URL = (
            'https://www.encodeproject.org/search/?'
            'type=file&'
            'format=json&'
            'limit=all&'
            'frame=object&'
            'file_format=bigWig'
        )
        exp_URL = (
            'https://www.encodeproject.org/search/?'
            'type=experiment&'
            'format=json&'
            'limit=all&'
            'frame=object'
        )

        # GET the object
        bw_response = requests.get(bw_URL, headers=HEADERS)
        exp_response = requests.get(exp_URL, headers=HEADERS)

        # Extract the JSON response as a python dict
        bw_response_json_dict = bw_response.json()
        exp_response_json_dict = exp_response.json()

        for entry in bw_response_json_dict['@graph']:
            self.bw_dict[entry['accession']] = entry
        for entry in exp_response_json_dict['@graph']:
            self.exp_dict[entry['accession']] = entry

    def associate_bw_to_exp(self):
        self.bw_to_exp = dict()

        for exp, entry in self.exp_dict.items():
            for _file in entry['files']:
                accession = _file.split('/')[2]
                if accession in self.bw_dict:
                    self.bw_to_exp[accession] = exp

    def create_experiment_entries(self):
        for exp, entry in self.exp_dict.items():
            has_bigwig = False
            for _file in entry['files']:
                accession = _file.split('/')[2]
                if accession in self.bw_dict:
                    has_bigwig = True
            if has_bigwig:
                self.experiments.append({
                    'name': exp,
                    'datasets': [],
                    'visible': True,
                    'controls': [],
                })

    def filter_and_organize_datasets(self):
        for i, experiment in enumerate(self.experiments):
            name = experiment['name']
            bw_files = []

            #  Get all bigwig files for an experiment
            for _file in self.exp_dict[name]['files']:
                accession = _file.split('/')[2]
                if accession in self.bw_dict:
                    bw_files.append(accession)

            #  Split by replicate, assembly, and strand
            replicates = dict()
            for _file in bw_files:
                output_type = self.bw_dict[_file]['output_type']

                replicate = tuple(self.bw_dict[_file]['biological_replicates'])
                assembly = self.bw_dict[_file]['assembly']
                if assembly in ASSEMBLY_ALIASES:
                    assembly = ASSEMBLY_ALIASES[assembly]
                key = (replicate, assembly)

                if key not in replicates:
                    replicates[key] = {
                        'plus': [],
                        'minus': [],
                        'ambiguous': [],
                    }
                if 'plus' in output_type:
                    replicates[key]['plus'].append(_file)
                elif 'minus' in output_type:
                    replicates[key]['minus'].append(_file)
                else:
                    replicates[key]['ambiguous'].append(_file)

            #  Filter replicate/strand files by selection level
            filtered = dict()
            for key, strand_files in replicates.items():
                filtered[key] = {
                    'plus': [],
                    'minus': [],
                    'ambiguous': [],
                }
                for strand, files in strand_files.items():
                    selection_level = len(SELECTION_HIERARCHY)

                    #  Find selection level
                    for _file in files:
                        output_type = self.bw_dict[_file]['output_type']
                        level = get_selection_level(output_type)
                        if level < selection_level:
                            selection_level = level

                    #  Get all bigwig files at selection level
                    keep_files = []
                    for _file in files:
                        output_type = self.bw_dict[_file]['output_type']
                        level = get_selection_level(output_type)
                        if level == selection_level:
                            keep_files.append(_file)

                    filtered[key][strand] = keep_files
            replicates = filtered

            #  Filter replicate/strand files by time
            filtered = dict()
            for key, strand_files in replicates.items():
                filtered[key] = {
                    'plus': [],
                    'minus': [],
                    'ambiguous': [],
                }
                for strand, files in strand_files.items():

                    #  Find most recent time
                    most_recent_time = None
                    for _file in files:
                        time = dateutil.parser.parse(
                            self.bw_dict[_file]['date_created']
                        ).replace(tzinfo=None)  # ENCODE inconsistent with TZs
                        if not most_recent_time:
                            most_recent_time = time
                        if time > most_recent_time:
                            most_recent_time = time

                    #  Get all bigwig files from most recent time
                    keep_files = []
                    for _file in files:
                        time = dateutil.parser.parse(
                            self.bw_dict[_file]['date_created']
                        ).replace(tzinfo=None)
                        if time == most_recent_time:
                            keep_files.append(_file)

                    filtered[key][strand] = keep_files
            replicates = filtered

            #  If stranded data exist, remove ambiguous
            for stranded_files in replicates.values():
                if stranded_files['plus'] and stranded_files['minus']:
                    stranded_files['ambiguous'] = []

            #  If merged, remove single replicates
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
                #  Create dataset entry
                replicate, assembly = key
                dataset = {
                    'replicate': replicate,
                    'assembly': assembly,
                }
                if strand_files['plus']:
                    dataset.update({'plus': strand_files['plus'][0]})
                if strand_files['minus']:
                    dataset.update({'minus': strand_files['minus'][0]})
                if strand_files['ambiguous']:
                    dataset.update({'ambiguous': strand_files['ambiguous'][0]})

                #  Append to experiment
                self.experiments[i]['datasets'].append(dataset)

    def filter_datasets_by_missing_strand_data(self):
        filtered_experiments = []
        for experiment in self.experiments:
            added = False
            for dataset in experiment['datasets']:
                if not (('plus' in dataset and 'minus' not in dataset) or
                        ('minus' in dataset and 'plus' not in dataset)):
                    if not added:
                        filtered_experiments.append({
                            'name': experiment['name'],
                            'datasets': [],
                            'visible': experiment['visible'],
                            'controls': experiment['controls'],
                        })
                        added = True
                    filtered_experiments[-1]['datasets'].append(dataset)
        self.experiments = filtered_experiments

    def set_empty_replicate_to_one(self):
        for experiment in self.experiments:
            for dataset in experiment['datasets']:
                if dataset['replicate'] is None:
                    dataset['replicate'] = (1,)

    def add_dataset_href_values(self):
        for experiment in self.experiments:
            for dataset in experiment['datasets']:
                if 'ambiguous' in dataset:
                    dataset['ambiguous_href'] = \
                        self.bw_dict[dataset['ambiguous']]['href']
                if 'plus' in dataset:
                    dataset['plus_href'] = \
                        self.bw_dict[dataset['plus']]['href']
                if 'minus' in dataset:
                    dataset['minus_href'] = \
                        self.bw_dict[dataset['minus']]['href']

    def make_control_associations(self):
        #  For ChIP-seq experiments, mark as 'low visibility' if a control
        for experiment in self.experiments:
            name = experiment['name']
            for control in self.exp_dict[name]['possible_controls']:
                experiment['controls'].append(control.split('/')[2])

    def mark_low_visibility_controls(self):
        #  For ChIP-seq experiments, mark as 'low visibility' if a control
        for experiment in self.experiments:
            name = experiment['name']
            if 'target' in self.exp_dict[name]:
                if 'Control' in self.exp_dict[name]['target'] and \
                        'ChIP-seq' in self.exp_dict[name]['assay_term_name']:
                    experiment['visible'] = False

    def add_detail(self):
        for experiment in self.experiments:
            experiment['detail'] = self.exp_dict[experiment['name']]
            for dataset in experiment['datasets']:
                if 'ambiguous' in dataset:
                    dataset['ambiguous_detail'] = \
                        self.bw_dict[dataset['ambiguous']]
                if 'plus' in dataset:
                    dataset['plus_detail'] = self.bw_dict[dataset['plus']]
                if 'minus' in dataset:
                    dataset['minus_detail'] = self.bw_dict[dataset['minus']]

    def get_experiments(self):
        self.call_api()
        self.associate_bw_to_exp()
        self.create_experiment_entries()
        self.filter_and_organize_datasets()
        self.filter_datasets_by_missing_strand_data()
        self.set_empty_replicate_to_one()
        self.add_dataset_href_values()
        self.make_control_associations()
        self.mark_low_visibility_controls()
        self.add_detail()

    def make_experiment_json(self, output_file):
        with open(output_file, 'w') as out:
            json.dump(self.experiments, out, indent=4)


@task()
def add_or_update_encode():

    def get_encode_url(url):
        return 'https://www.encodeproject.org{}'.format(url)

    project = models.Project.objects.get_or_create(
        name='ENCODE',
    )[0]

    encode = Encode()
    encode.get_experiments()
    print('{} experiments found in ENCODE!!'.format(len(encode.experiments)))

    datasets_to_process = set()

    for experiment in encode.experiments:
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
            consortial_id=experiment['name'],
            defaults={
                'name': experiment_name,
                'project': project,
                'description': experiment_description,
                'experiment_type': experiment_type,
                'cell_type': biosample_term_name,
                'slug': experiment['name'],
            },
        )
        if target:
            exp.target = target
            exp.save()

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
                consortial_id = dataset['ambiguous']
            try:
                plus_url = get_encode_url(dataset['plus_href'])
                minus_url = get_encode_url(dataset['minus_href'])
            except:
                plus_url = None
                minus_url = None
            else:
                consortial_id = '{}-{}'.format(
                    dataset['plus'],
                    dataset['minus'],
                )

            # Create dataset name
            assembly = dataset['assembly']
            dataset_name = '{}-{}-{}'.format(
                consortial_id,
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
                    consortial_id=consortial_id,
                    experiment=exp,
                    defaults={
                        'name': dataset_name,
                        'assembly': assembly_obj,
                        'slug': consortial_id,
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
                    ds.processed = False
                    ds.save()

                if not ds.processed:
                    datasets_to_process.add(ds)

    print('Processing {} datasets...'.format(len(datasets_to_process)))
    process_datasets(list(datasets_to_process))
