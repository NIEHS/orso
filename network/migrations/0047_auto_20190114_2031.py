# -*- coding: utf-8 -*-
# Generated by Django 1.10.1 on 2019-01-15 01:31
from __future__ import unicode_literals

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('network', '0046_auto_20180827_1742'),
    ]

    operations = [
        migrations.AlterUniqueTogether(
            name='experiment',
            unique_together=set([('project', 'consortial_id', 'experiment_type', 'cell_type', 'target')]),
        ),
    ]