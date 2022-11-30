# Generated by Django 4.1.3 on 2022-11-30 07:17

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('products', '0001_initial'),
        ('susers', '0001_initial'),
        ('deliveries', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Order',
            fields=[
                ('orders_id', models.AutoField(primary_key=True, serialize=False)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('delivery', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='deliveries.delivery')),
                ('product', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='products.products')),
                ('shop_user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='susers.susers')),
            ],
            options={
                'db_table': 'shop_orders',
            },
        ),
    ]