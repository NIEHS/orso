# -*- coding: utf-8 -*-
# Generated by Django 1.10.1 on 2018-07-30 14:40
from __future__ import unicode_literals

from django.db import migrations

CREATE_TWO_PARTIAL_INDEX = """
    CREATE UNIQUE INDEX similarity_uni_idx_1
    ON network_similarity (sim_type, experiment_1_id, experiment_2_id, dataset_1_id, dataset_2_id)
    WHERE network_similarity.dataset_1_id IS NOT NULL AND network_similarity.dataset_2_id IS NOT NULL;

    CREATE UNIQUE INDEX similarity_uni_idx_2
    ON network_similarity (sim_type, experiment_1_id, experiment_2_id)
    WHERE network_similarity.dataset_1_id IS NULL AND network_similarity.dataset_2_id IS NULL;
"""

DROP_TWO_PARTIAL_INDEX = """
    DROP INDEX similarity_uni_idx_1;
    DROP INDEX similarity_uni_idx_2;
"""


class Migration(migrations.Migration):

    dependencies = [
        ('network', '0042_auto_20180716_1545'),
    ]

    operations = [
        migrations.AlterUniqueTogether(
            name='similarity',
            unique_together=set([]),
        ),
        migrations.RunSQL(CREATE_TWO_PARTIAL_INDEX, DROP_TWO_PARTIAL_INDEX)
    ]
