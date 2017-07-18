from subprocess import call

BIGWIG_AVERAGE_OVER_BED_PATH = \
    '/Users/lavenderca/Downloads/bigWigAverageOverBed'


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
