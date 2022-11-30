from django.db import models

class Susers(models.Model):
    use_in_migrations = True

    product_id = models.AutoField(primary_key=True)
    name = models.TextField()
    price = models.TextField()
    image_url = models.TextField()

    class Meta:
        db_table = "shop_susers"

    def __str__(self):
        return f"{self.pk} {self.name} {self.price} {self.image_url}"
