from django.db import models

from shop.products.models import Products
from shop.susers.models import Susers


class Carts(models.Model):
    use_in_migrations = True

    cart_id = models.AutoField(primary_key=True)

    product = models.ForeignKey(Products, on_delete=models.CASCADE)
    shop_user = models.ForeignKey(Susers, on_delete=models.CASCADE)

    class Meta:
        db_table = "shop_carts"

    def __str__(self):
        return f"{self.pk}"
