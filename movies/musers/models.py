from django.db import models


class Musers(models.Model):
    use_in_migrations = True

    musers_id = models.AutoField(primary_key=True)
    email = models.TextField()
    nickname = models.TextField()
    password = models.TextField()
    age = models.TextField()



    class Meta:
        db_table = "movies_musers"

    def __str__(self):
        return f"{self.pk} {self.email} {self.nickname} {self.password} {self.age}"