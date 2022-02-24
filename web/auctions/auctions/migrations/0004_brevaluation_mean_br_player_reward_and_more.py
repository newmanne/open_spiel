# Generated by Django 4.0.2 on 2022-02-23 23:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('auctions', '0003_alter_bestresponse_options_remove_experiment_game_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='brevaluation',
            name='mean_br_player_reward',
            field=models.FloatField(default=0),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='bestresponse',
            name='model',
            field=models.BinaryField(null=True),
        ),
    ]
