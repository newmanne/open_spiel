# Generated by Django 4.0.2 on 2023-06-26 21:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('auctions', '0005_evaluation_nash_conv_runtime'),
    ]

    operations = [
        migrations.AlterField(
            model_name='evaluation',
            name='nash_conv_runtime',
            field=models.FloatField(null=True),
        ),
    ]