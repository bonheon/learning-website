from djongo import models

class Data(models.Model):
    timestamp = models.DateTimeField()
    p13 = models.FloatField()
    edge = models.FloatField()
    exed = models.FloatField()

    def __str__(self):
        return f"{self.timestamp}"