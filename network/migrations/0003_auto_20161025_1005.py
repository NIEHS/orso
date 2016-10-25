# -*- coding: utf-8 -*-
# Generated by Django 1.10.2 on 2016-10-25 14:05
from __future__ import unicode_literals

import django.contrib.postgres.fields
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('network', '0002_auto_20161024_1714'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='datarecommendation',
            options={'ordering': ('score', '-last_updated')},
        ),
        migrations.AlterModelOptions(
            name='userrecommendation',
            options={'ordering': ('score', '-last_updated')},
        ),
        migrations.RemoveField(
            model_name='gene',
            name='names',
        ),
        migrations.AddField(
            model_name='gene',
            name='aliases',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.CharField(max_length=64), default=['alias'], size=None),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='geneannotation',
            name='name',
            field=models.CharField(default='name', max_length=32),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='genomicregions',
            name='name',
            field=models.CharField(default='name', max_length=32),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='geneannotation',
            name='enhancers',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='enhancers', to='network.GenomicRegions'),
        ),
        migrations.AlterField(
            model_name='geneannotation',
            name='gtf_file',
            field=models.FileField(upload_to=''),
        ),
        migrations.AlterField(
            model_name='geneannotation',
            name='promoters',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='promoters', to='network.GenomicRegions'),
        ),
        migrations.AlterField(
            model_name='genomeassembly',
            name='default_annotation',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='network.GeneAnnotation'),
        ),
        migrations.AlterField(
            model_name='myuser',
            name='favorite_data',
            field=models.ManyToManyField(blank=True, to='network.Dataset'),
        ),
        migrations.AlterField(
            model_name='myuser',
            name='favorite_users',
            field=models.ManyToManyField(blank=True, to='network.MyUser'),
        ),
    ]
