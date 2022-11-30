from django.db import models

class Busers(models.Model):
    use_in_migrations = True

    blog_id = models.IntegerField(primary_key=True)
    email = models.TextField()
    nickname = models.TextField()
    password = models.TextField()

    class Meta:
        db_table = "blog_busers"

    def __str__(self):
        return f"{self.pk} {self.content} {self.nickname} {self.password}"
