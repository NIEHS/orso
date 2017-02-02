#!/usr/bin/env python

import requests
import dateutil.parser
import click
import json

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


@click.command()
@click.argument('json_output')
def cli(json_output):
    '''
    Interact with the ENCODE REST API.
    '''

    encode = Encode()
    encode.get_experiments()
    encode.make_experiment_json(json_output)

if __name__ == '__main__':
    cli()
