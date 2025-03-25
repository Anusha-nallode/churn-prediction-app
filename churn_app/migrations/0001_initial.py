# Generated by Django 5.1.7 on 2025-03-24 15:15

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Customer",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("customer_id", models.IntegerField(unique=True)),
                ("age", models.IntegerField()),
                ("balance", models.FloatField()),
                ("estimated_salary", models.FloatField()),
                ("churn_prediction", models.CharField(max_length=10)),
            ],
        ),
    ]
