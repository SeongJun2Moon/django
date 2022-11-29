from django.db import models


class M_users(models.Model):
    use_in_migrations = True

    id = models.IntegerField(primary_key=True)
    email = models.CharField(max_length=120, unique=True)
    nickname = models.CharField(max_length=20, unique=True)
    password = models.CharField(max_length=255)
    age = models.IntegerField()

    class Meta:
        db_table = "m_users"

    def __str__(self):
        return f"{self.pk} {self.nickname}"