# Generated by Django 4.0.2 on 2022-09-30 22:55

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('auctions', '0006_remove_brevaluation_id_remove_evaluation_id_and_more'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='experiment',
            options={'ordering': ('created',)},
        ),
        migrations.AddField(
            model_name='equilibriumsolverrun',
            name='generation',
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AddField(
            model_name='equilibriumsolverrun',
            name='parent',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='auctions.equilibriumsolverruncheckpoint'),
        ),
    ]
