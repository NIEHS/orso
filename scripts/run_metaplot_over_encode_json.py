from analysis.metaplot import MetaPlot
from collections import defaultdict
from datetime import datetime
import json
import click
import os
import tempfile
import multiprocessing
import urllib.request
import shutil

BED_DICT = None
OUTPUT_DIR = None


def get_encode_url(url):
    return 'https://www.encodeproject.org{}'.format(url)


def read(bed_list):
    bed_dict = defaultdict(list)
    with open(bed_list) as f:
        for line in f:
            _file, assembly = line.strip().split()
            bed_dict[assembly].append({
                'file': _file,
                'name': os.path.splitext(os.path.basename(_file))[0],
            })
    return bed_dict


def process_dataset(dataset):
    assembly = dataset['assembly']
    bed_files = BED_DICT[assembly]

    run = False
    if 'ambiguous_href' in dataset:
        dataset_name = dataset['ambiguous']
        for bed in bed_files:
            out_header = '{}.{}'.format(
                dataset_name,
                bed['name']
            )
            intersection_file = os.path.join(
                OUTPUT_DIR, out_header + '.intersection.json')
            metaplot_file = os.path.join(
                OUTPUT_DIR, out_header + '.metaplot.json')
            if not (os.path.isfile(intersection_file) and
                    os.path.isfile(metaplot_file)):
                run = True
    else:
        dataset_f = dataset['plus']
        dataset_r = dataset['minus']
        dataset_name = '{}.{}'.format(dataset_f, dataset_r)
        for bed in bed_files:
            out_header = '{}.{}.{}'.format(
                dataset_f,
                dataset_r,
                bed['name']
            )
            intersection_file = os.path.join(
                OUTPUT_DIR, out_header + '.intersection.json')
            metaplot_file = os.path.join(
                OUTPUT_DIR, out_header + '.metaplot.json')
            if not (os.path.isfile(intersection_file) and
                    os.path.isfile(metaplot_file)):
                run = True

    if not run:
        print('{}: files found; skipped.'.format(dataset_name))
    else:
        if 'ambiguous_href' in dataset:
            print('{}: processing {}.'.format(
                str(datetime.now()),
                dataset['ambiguous'],
            ))
            url = get_encode_url(dataset['ambiguous_href'])
            dataset_name = dataset['ambiguous']
            temp_bw = tempfile.NamedTemporaryFile()
            print('{}: downloading {}.'.format(
                str(datetime.now()),
                dataset['ambiguous'],
            ))
            with urllib.request.urlopen(url) as response:
                shutil.copyfileobj(response, temp_bw)
            print('{}: {} download complete.'.format(
                str(datetime.now()),
                dataset['ambiguous'],
            ))
            for bed in bed_files:
                print('{}: processing {} over {}.'.format(
                    str(datetime.now()),
                    dataset['ambiguous'],
                    bed['name'],
                ))
                meta = MetaPlot(bed['file'], single_bw=temp_bw.name)
                out_header = '{}.{}'.format(
                    dataset_name,
                    bed['name']
                )
                try:
                    meta.create_intersection_json(os.path.join(
                        OUTPUT_DIR, out_header + '.intersection.json'))
                    meta.create_metaplot_json(os.path.join(
                        OUTPUT_DIR, out_header + '.metaplot.json'))
                except:
                    print('{}: processing {} over {} failed.'.format(
                        str(datetime.now()),
                        dataset['ambiguous'],
                        bed['name'],
                    ))
                else:
                    print('{}: processing {} over {} complete.'.format(
                        str(datetime.now()),
                        dataset['ambiguous'],
                        bed['name'],
                    ))
            temp_bw.close()
            print('{}: processing {} complete.'.format(
                str(datetime.now()),
                dataset['ambiguous'],
            ))
        else:
            print('{}: processing {}/{}.'.format(
                str(datetime.now()),
                dataset['plus'],
                dataset['minus'],
            ))
            url_f = get_encode_url(dataset['plus_href'])
            url_r = get_encode_url(dataset['minus_href'])
            dataset_f = dataset['plus']
            dataset_r = dataset['minus']
            temp_bw_f = tempfile.NamedTemporaryFile()
            temp_bw_r = tempfile.NamedTemporaryFile()

            print('{}: downloading {}.'.format(
                str(datetime.now()),
                dataset['plus'],
            ))
            with urllib.request.urlopen(url_f) as response:
                shutil.copyfileobj(response, temp_bw_f)
            print('{}: {} download complete.'.format(
                str(datetime.now()),
                dataset['plus'],
            ))

            print('{}: downloading {}.'.format(
                str(datetime.now()),
                dataset['minus'],
            ))
            with urllib.request.urlopen(url_r) as response:
                shutil.copyfileobj(response, temp_bw_r)
            print('{}: {} download complete.'.format(
                str(datetime.now()),
                dataset['minus'],
            ))

            for bed in bed_files:
                print('{}: processing {}/{} over {}.'.format(
                    str(datetime.now()),
                    dataset['plus'],
                    dataset['minus'],
                    bed['name'],
                ))
                meta = MetaPlot(bed['file'], paired_1_bw=temp_bw_f.name,
                                paired_2_bw=temp_bw_r.name)
                out_header = '{}.{}.{}'.format(
                    dataset_f,
                    dataset_r,
                    bed['name']
                )
                try:
                    meta.create_intersection_json(os.path.join(
                        OUTPUT_DIR, out_header + '.intersection.json'))
                    meta.create_metaplot_json(os.path.join(
                        OUTPUT_DIR, out_header + '.metaplot.json'))
                except:
                    print('{}: processing {}/{} over {} failed.'.format(
                        str(datetime.now()),
                        dataset['plus'],
                        dataset['minus'],
                        bed['name'],
                    ))
                else:
                    print('{}: processing {}/{} over {} complete.'.format(
                        str(datetime.now()),
                        dataset['plus'],
                        dataset['minus'],
                        bed['name'],
                    ))
            temp_bw_f.close()
            temp_bw_r.close()
            print('{}: processing {}/{} complete.'.format(
                str(datetime.now()),
                dataset['plus'],
                dataset['minus'],
            ))


@click.command()
@click.argument('json_input')
@click.argument('bed_list')
@click.argument('output_directory')
@click.option('--processes', default=1, help='Processes to run in parallel',
              type=int)
def cli(json_input, bed_list, output_directory, processes):
    '''
    Create metaplot files for each entry in JSON for specified BED files
    '''
    with open(json_input) as _in:
        experiments = json.load(_in)

    global BED_DICT
    global OUTPUT_DIR
    BED_DICT = read(bed_list)
    OUTPUT_DIR = output_directory

    dataset_list = []
    for experiment in experiments:
        for dataset in experiment['datasets']:
            dataset_list.append(dataset)
    p = multiprocessing.Pool(processes)
    p.map(process_dataset, dataset_list)

if __name__ == '__main__':
    cli()
