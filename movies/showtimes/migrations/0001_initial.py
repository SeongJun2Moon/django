# Generated by Django 4.1.3 on 2022-11-30 07:17

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('cinema', '0001_initial'),
        ('theater', '0001_initial'),
        ('movies', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Showtimes',
            fields=[
                ('showtimes_id', models.AutoField(primary_key=True, serialize=False)),
                ('start_time', models.DateTimeField()),
                ('end_time', models.DateTimeField()),
                ('cinema', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='cinema.cinema')),
                ('movie', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='movies.movie')),
                ('theater', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='theater.theaters')),
            ],
            options={
                'db_table': 'movies_showtimes',
            },
        ),
    ]