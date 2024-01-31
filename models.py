from django.db import models

class Picture(models.Model):
    title = models.CharField(max_length=255)
    image = models.ImageField(upload_to='images/')
    category = models.CharField(max_length=50, default='default_category')

    def __str__(self):
        return self.title

class Patient(models.Model):
    age = models.IntegerField()
    alcohol_usage = models.BooleanField()
    bmi = models.FloatField()
    cigarettes_usage = models.BooleanField()
    had_covid = models.BooleanField()
    depression = models.BooleanField()
    diabetes = models.BooleanField()
    diabetic = models.BooleanField()
    physical_activity = models.BooleanField()
    sleep_time = models.FloatField()
    diff_walking = models.BooleanField()
    gen_health = models.BooleanField()
    alcohol_drinking = models.BooleanField()
    smoking = models.BooleanField()
    heart_disease = models.BooleanField()
    stroke = models.BooleanField()
    physical_health = models.IntegerField()
    mental_health = models.IntegerField()
    sex = models.CharField(max_length=6, choices=[('female', 'Female'), ('male', 'Male')])
    age_category = models.CharField(max_length=10, choices=[('0-18', '0-18'), ('19-30', '19-30'), ('31-50', '31-50'), ('51-70', '51-70'), ('71+', '71+')])
    race = models.CharField(max_length=20, choices=[('White', 'White'), ('Black', 'Black'), ('Asian', 'Asian'), ('Hispanic', 'Hispanic'), ('Other', 'Other')])
    asthma = models.BooleanField()
    kidney_disease = models.BooleanField()
    skin_cancer = models.BooleanField()
