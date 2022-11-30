from django.db import models

from blog.busers.models import Busers
from blog.posts.models import Posts


class Views(models.Model):
    use_in_migrations = True

    views_id = models.AutoField(primary_key=True)
    ip_address = models.TextField()
    create_at = models.DateTimeField(auto_now_add=True)

    blog_user = models.ForeignKey(Busers, on_delete=models.CASCADE)
    post = models.ForeignKey(Posts, on_delete=models.CASCADE)

    class Meta:
        db_table = "blog_views"

    def __str__(self):
        return f"{self.pk} {self.ip_address} {self.create_at}"