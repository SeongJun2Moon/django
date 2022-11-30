from django.db import models

class Cinema(models.Model):
    use_in_migrations = True

    cinema_id = models.AutoField(primary_key=True)
    title = models.TextField()
    image_url = models.TextField()
    address = models.TextField()
    detail_address = models.TextField()

    class Meta:
        db_table = "movies_cinema"

    def __str__(self):
        return f"{self.pk} {self.title} {self.image_url} {self.address} {self.detail_address}"