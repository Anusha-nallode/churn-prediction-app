from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("login/", views.login_view, name="login"),
    path("register/", views.register, name="register"),
    path("predict/", views.predict_churn, name="predict_churn"),
    path("logout/", views.logout_view, name="logout"),
]
