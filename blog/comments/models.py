from django.db import models

from blog.busers.models import Busers
from blog.posts.models import Posts


class Comments(models.Model):
    use_in_migrations = True

    comments_id = models.IntegerField(primary_key=True)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True) #datatime_start 장고
    updatated_at = models.DateTimeField(auto_now=True) #datatime_end 장고
    parent_id = models.TextField(null=True)

    blog_user = models.ForeignKey(Busers, on_delete=models.CASCADE)
    post = models.ForeignKey(Posts, on_delete=models.CASCADE)


    class Meta:
        db_table = "blog_comments"

    def __str__(self):
        return f"{self.pk} {self.content} {self.created_at} {self.updatated_at} {self.parent_id}"