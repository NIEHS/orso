import os
from subprocess import call

BIGWIG_AVERAGE_OVER_BED_PATH = \
    os.path.abspath(os.path.join(
        os.path.dirname(__file__), os.pardir, 'bin', 'bigWigAverageOverBed'))


def call_bigwig_average_over_bed(bigwig_name, bed_name, out_name):
    '''
    Call Kent tools bigWigAverageOverBed.
    '''
    call([
        BIGWIG_AVERAGE_OVER_BED_PATH,
        bigwig_name,
        bed_name,
        out_name,
    ])
