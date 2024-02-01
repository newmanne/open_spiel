# Generated by Django 4.0.2 on 2023-06-23 21:03

import django.contrib.postgres.fields
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('auctions', '0002_evaluation_nash_conv'),
    ]

    operations = [
        migrations.AddField(
            model_name='evaluation',
            name='player_improvements',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), default=[], size=None),
            preserve_default=False,
        ),
    ]