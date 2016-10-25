# -*- coding: utf-8 -*-
# Generated by Django 1.10.2 on 2016-10-24 21:11
from __future__ import unicode_literals

from django.conf import settings
import django.contrib.postgres.fields.jsonb
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='CorrelationCell',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('score', models.FloatField()),
                ('last_updated', models.DateTimeField(auto_now=True)),
            ],
        ),
        migrations.CreateModel(
            name='DataRecommendation',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('last_updated', models.DateTimeField(auto_now=True)),
                ('score', models.FloatField()),
            ],
            options={
                'ordering': ('-last_updated',),
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='Dataset',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('data_type', models.CharField(choices=[('Cage', 'Cage'), ('ChiaPet', 'ChiaPet'), ('ChipSeq', 'ChipSeq'), ('DnaseDgf', 'DnaseDgf'), ('DnaseSeq', 'DnaseSeq'), ('FaireSeq', 'FaireSeq'), ('Mapability', 'Mapability'), ('Nucleosome', 'Nucleosome'), ('Orchid', 'Orchid'), ('RepliChip', 'RepliChip'), ('RepliSeq', 'RepliSeq'), ('RipSeq', 'RipSeq'), ('RnaPet', 'RnaPet'), ('RnaSeq', 'RnaSeq'), ('StartSeq', 'StartSeq'), ('Other', 'Other (describe in "description" field)')], max_length=16)),
                ('cell_type', models.CharField(max_length=128)),
                ('antibody', models.CharField(max_length=128)),
                ('description', models.TextField(blank=True)),
                ('ambiguous_url', models.URLField()),
                ('plus_url', models.URLField()),
                ('minus_url', models.URLField()),
                ('name', models.CharField(max_length=128)),
                ('slug', models.CharField(max_length=128)),
                ('intersection_values', django.contrib.postgres.fields.jsonb.JSONField()),
                ('created', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='Exon',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('id_num', models.IntegerField()),
                ('start', models.IntegerField()),
                ('end', models.IntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='Gene',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('names', django.contrib.postgres.fields.jsonb.JSONField()),
                ('chromosome', models.CharField(max_length=32)),
                ('strand', models.CharField(choices=[('+', '+'), ('-', '-')], max_length=1)),
                ('start', models.IntegerField()),
                ('end', models.IntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='GeneAnnotation',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('gtf_file', models.FileField(upload_to='')),
                ('last_updated', models.DateTimeField(auto_now=True)),
            ],
        ),
        migrations.CreateModel(
            name='GenomeAssembly',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=32, unique=True)),
                ('last_updated', models.DateTimeField(auto_now=True)),
                ('default_annotation', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='network.GeneAnnotation')),
            ],
        ),
        migrations.CreateModel(
            name='GenomicRegions',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('bed_file', models.FileField(upload_to='')),
                ('last_updated', models.DateTimeField(auto_now=True)),
                ('assembly', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='network.GenomeAssembly')),
            ],
        ),
        migrations.CreateModel(
            name='MetaPlot',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('relative_start', models.IntegerField()),
                ('relative_end', models.IntegerField()),
                ('meta_plot', django.contrib.postgres.fields.jsonb.JSONField()),
                ('last_updated', models.DateTimeField(auto_now=True)),
                ('genomic_regions', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='network.GenomicRegions')),
            ],
        ),
        migrations.CreateModel(
            name='MyUser',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('slug', models.CharField(max_length=128)),
                ('favorite_data', models.ManyToManyField(to='network.Dataset')),
                ('favorite_users', models.ManyToManyField(to='network.MyUser')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='UserRecommendation',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('last_updated', models.DateTimeField(auto_now=True)),
                ('score', models.FloatField()),
                ('owner', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='network.MyUser')),
                ('recommended', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='recommended', to='network.MyUser')),
            ],
            options={
                'ordering': ('-last_updated',),
                'abstract': False,
            },
        ),
        migrations.AddField(
            model_name='geneannotation',
            name='assembly',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='network.GenomeAssembly'),
        ),
        migrations.AddField(
            model_name='geneannotation',
            name='enhancers',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='enhancers', to='network.GenomicRegions'),
        ),
        migrations.AddField(
            model_name='geneannotation',
            name='promoters',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='promoters', to='network.GenomicRegions'),
        ),
        migrations.AddField(
            model_name='gene',
            name='annotation',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='network.GeneAnnotation'),
        ),
        migrations.AddField(
            model_name='exon',
            name='gene',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='network.Gene'),
        ),
        migrations.AddField(
            model_name='dataset',
            name='enhancer_metaplot',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='enhancer_meta', to='network.MetaPlot'),
        ),
        migrations.AddField(
            model_name='dataset',
            name='owners',
            field=models.ManyToManyField(to='network.MyUser'),
        ),
        migrations.AddField(
            model_name='dataset',
            name='promoter_metaplot',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='promoter_meta', to='network.MetaPlot'),
        ),
        migrations.AddField(
            model_name='datarecommendation',
            name='owner',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='network.MyUser'),
        ),
        migrations.AddField(
            model_name='datarecommendation',
            name='recommended',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='network.Dataset'),
        ),
        migrations.AddField(
            model_name='correlationcell',
            name='x_dataset',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='x', to='network.Dataset'),
        ),
        migrations.AddField(
            model_name='correlationcell',
            name='y_dataset',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='y', to='network.Dataset'),
        ),
        migrations.AlterUniqueTogether(
            name='metaplot',
            unique_together=set([('genomic_regions', 'relative_start', 'relative_end')]),
        ),
        migrations.AlterUniqueTogether(
            name='correlationcell',
            unique_together=set([('x_dataset', 'y_dataset')]),
        ),
    ]
