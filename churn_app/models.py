from django.db import models

class Customer(models.Model):
    customer_id = models.IntegerField(unique=True)
    age = models.IntegerField()
    balance = models.FloatField()
    estimated_salary = models.FloatField()
    churn_prediction = models.CharField(max_length=10)

    def __str__(self):
        return f"Customer {self.customer_id}"
