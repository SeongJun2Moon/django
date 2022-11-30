from django.db import models

class Categories(models.Model):
    use_in_migrations = True

    categories_id = models.AutoField(primary_key=True)
    name = models.TextField()

    class Meta:
        db_table = "shop_categories"

    def __str__(self):
        return f"{self.pk} {self.name}"
