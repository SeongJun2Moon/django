from django.db import models

from movies.cinema.models import Cinema
from movies.movies.models import Movie
from movies.theater.models import Theaters


class Showtimes(models.Model):
    use_in_migrations = True

    showtimes_id = models.AutoField(primary_key=True)
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()

    cinema = models.ForeignKey(Cinema, on_delete=models.CASCADE)
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
    theater = models.ForeignKey(Theaters, on_delete=models.CASCADE)

    class Meta:
        db_table = "movies_showtimes"

    def __str__(self):
        return f"{self.pk} {self.start_time} {self.end_time}"