from django.db import models

from shop.categories.models import Categories


class Products(models.Model):
    use_in_migrations = True

    products_id = models.AutoField(primary_key=True)
    name = models.TextField()
    price = models.IntegerField()
    image_url = models.TextField()

    category = models.ForeignKey(Categories, on_delete=models.CASCADE)


    class Meta:
        db_table = "shop_products"

    def __str__(self):
        return f"{self.pk} {self.name} {self.price} {self.image_url}"
