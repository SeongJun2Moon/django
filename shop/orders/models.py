from django.db import models

from shop.deliveries.models import Delivery
from shop.products.models import Products
from shop.susers.models import Susers


class Order(models.Model):
    use_in_migrations = True

    orders_id = models.AutoField(primary_key=True)
    created_at = models.DateTimeField(auto_now_add=True)

    product = models.ForeignKey(Products, on_delete=models.CASCADE)
    shop_user = models.ForeignKey(Susers, on_delete=models.CASCADE)
    delivery = models.ForeignKey(Delivery, on_delete=models.CASCADE)

    class Meta:
        db_table = "shop_orders"

    def __str__(self):
        return f"{self.pk} {self.created_at}"
