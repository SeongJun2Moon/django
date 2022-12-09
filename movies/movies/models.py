from django.db import models

class Movie(models.Model):
    use_in_migrations = True

    movies_id = models.AutoField(primary_key=True)
    title = models.TextField()
    director = models.TextField()
    description = models.TextField()
    poster_url = models.TextField()
    running_time = models.TextField()
    age_rate = models.IntegerField()

    class Meta:
        db_table = "movies_movie"

    def __str__(self):
        return f"{self.pk} {self.title} {self.director} {self.description} {self.poster_url}" \
               f"{self.running_time} {self.age_rate}"


