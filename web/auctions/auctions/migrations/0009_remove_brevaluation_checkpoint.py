# Generated by Django 4.0.2 on 2022-12-03 02:24

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('auctions', '0008_brevaluation_checkpoint'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='brevaluation',
            name='checkpoint',
        ),
    ]