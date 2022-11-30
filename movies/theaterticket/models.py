from django.db import models

from movies.musers.models import Musers
from movies.showtimes.models import Showtimes
from movies.theater.models import Theaters


class Theater_tickets(models.Model):
    use_in_migrations = True

    teaher_tickets_id = models.AutoField(primary_key=True)
    x = models.IntegerField()
    y = models.IntegerField()

    showtimes = models.ForeignKey(Showtimes, on_delete=models.CASCADE)
    theaters = models.ForeignKey(Theaters, on_delete=models.CASCADE)
    movie_user = models.ForeignKey(Musers, on_delete=models.CASCADE)

    class Meta:
        db_table = "movies_theater_tickets"

    def __str__(self):
        return f"{self.pk} {self.x} {self.y}"