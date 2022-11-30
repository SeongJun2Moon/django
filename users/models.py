from django.db import models


class User(models.Model):
    use_in_migrations = True

    id = models.AutoField(primary_key=True) #auto_increment 자동설정
    username = models.CharField(max_length=100, unique=True)
    password = models.CharField(max_length=255)
    created_at = models.DateTimeField()
    rank = models.IntegerField(default=1)
    point = models.IntegerField(default=0)

    class Meta:
        db_table = "users"

    def __str__(self):
        return f"{self.pk} {self.username}"

