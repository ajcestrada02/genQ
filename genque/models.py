from django.db import models

# Create your models here.
class Generate(models.Model):
	title = models.CharField(max_length=250)
	text = models.TextField()
	MCQ = models.BooleanField()
	MCQ_items = models.SmallIntegerField()
	TF = models.BooleanField()
	TF_items = models.SmallIntegerField()
	FINB = models.BooleanField()
	FINB_items = models.SmallIntegerField()
	
class Statistics(models.Model):
	report = models.ForeignKey(Generate, on_delete=models.CASCADE)
	rank = models.CharField(max_length=250)
	topic = models.CharField(max_length=250)
	stats = models.CharField(max_length=250)




