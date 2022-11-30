from django.db import models

from blog.busers.models import Busers


class Posts(models.Model):
    use_in_migrations = True

    posts_id = models.IntegerField(primary_key=True)
    title = models.TextField()
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updatated_at = models.DateTimeField(auto_now=True)

    blog_user = models.ForeignKey(Busers, on_delete=models.CASCADE)

    class Meta:
        db_table = "blog_posts"

    def __str__(self):
        return f"{self.pk} {self.title} {self.content} {self.created_at} {self.updatated_at}"