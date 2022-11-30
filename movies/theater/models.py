from django.db import models

from movies.cinema.models import Cinema


class Theaters(models.Model):
    use_in_migrations = True

    theaters_id = models.AutoField(primary_key=True)
    title = models.TextField()
    seat = models.TextField()

    cinema = models.ForeignKey(Cinema, on_delete=models.CASCADE)

    class Meta:
        db_table = "movies_theaters"

    def __str__(self):
        return f"{self.pk} {self.title} {self.seat}"