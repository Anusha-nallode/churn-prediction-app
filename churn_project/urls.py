from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("churn_app.urls")),  # This links to `churn_app/urls.py`
    
]
