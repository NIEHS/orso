from network.tasks.analysis.metaplot import MetaPlot
from collections import defaultdict
from datetime import datetime
import json
import click
import os
import multiprocessing
from subprocess import call

BED_DICT = None
OUTPUT_DIR = None
WORKING_DIR = os.getcwd()


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

    if 'ambiguous_href' in dataset:
        dataset_name = dataset['ambiguous']

        print('{}: processing {}.'.format(
            str(datetime.now()),
            dataset['ambiguous'],
        ))

        for bed in bed_files:
            print('{}: processing {} over {}.'.format(
                str(datetime.now()),
                dataset['ambiguous'],
                bed['name'],
            ))
            meta = MetaPlot(bed['file'], single_bw=os.path.join(
                OUTPUT_DIR, dataset['ambiguous'] + '.bigWig'))
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

        print('{}: processing {} complete.'.format(
            str(datetime.now()),
            dataset['ambiguous'],
        ))
    else:
        dataset_f = dataset['plus']
        dataset_r = dataset['minus']
        dataset_name = '{}.{}'.format(dataset_f, dataset_r)

        print('{}: processing {}/{}.'.format(
            str(datetime.now()),
            dataset['plus'],
            dataset['minus'],
        ))

        for bed in bed_files:
            print('{}: processing {}/{} over {}.'.format(
                str(datetime.now()),
                dataset['plus'],
                dataset['minus'],
                bed['name'],
            ))
            meta = MetaPlot(
                bed['file'],
                paired_1_bw=os.path.join(
                    OUTPUT_DIR, dataset['plus'] + '.bigWig'),
                paired_2_bw=os.path.join(
                    OUTPUT_DIR, dataset['minus'] + '.bigWig'),
            )
            out_header = '{}.{}.{}'.format(
                dataset_f,
                dataset_r,
                bed['name'],
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
            assembly = dataset['assembly']
            bed_files = BED_DICT[assembly]

            run = False

            if 'ambiguous_href' in dataset:
                dataset_name = dataset['ambiguous']
            else:
                dataset_f = dataset['plus']
                dataset_r = dataset['minus']
                dataset_name = '{}.{}'.format(dataset_f, dataset_r)

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
            if run:
                dataset_list.append(dataset)
            else:
                print('{}: files found; skipped.'.format(dataset_name))

    for i in range(0, len(dataset_list), 100):
        if i + 100 <= len(dataset_list):
            dataset_chunk = dataset_list[i:i + 100]
        else:
            dataset_chunk = dataset_list[i:len(dataset_list)]

        chunk_files = []
        for field in ['ambiguous_href', 'plus_href', 'minus_href']:
            for dataset in dataset_chunk:
                if field in dataset:
                    chunk_files.append(dataset[field])

        download_list_path = os.path.join(OUTPUT_DIR, 'download_list.txt')
        download_list_file = open(download_list_path, 'w')

        for fn in chunk_files:
            download_list_file.write('{}\n'.format(get_encode_url(fn)))

        download_list_file.close()

        os.chdir(OUTPUT_DIR)
        call([
            'aria2c', '-x', '16', '-s', '16', '-i',
            os.path.basename(download_list_path),
        ])
        os.chdir(WORKING_DIR)

        p = multiprocessing.Pool(processes)
        p.map(process_dataset, dataset_chunk)
        p.close()
        p.join()

        os.chdir(OUTPUT_DIR)
        for fn in chunk_files:
            os.remove(os.path.basename(fn))
        os.chdir(WORKING_DIR)


if __name__ == '__main__':
    cli()
