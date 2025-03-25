import os
from pathlib import Path
from django.core.management.utils import get_random_secret_key

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Security: SECRET_KEY (Ensure this key is kept secret in production)
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", get_random_secret_key())

# Debug mode (Set False in production)
DEBUG = True

# Allowed hosts (Ensure this is updated in production)
ALLOWED_HOSTS = ["localhost", "127.0.0.1"]

# Installed applications
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "churn_app",  # Your app
]
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]
# Middleware
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

# URL configuration
ROOT_URLCONF = "churn_project.urls"

import os

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / "churn_app/templates"],  # Ensure templates are in the correct directory
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]



# WSGI application
WSGI_APPLICATION = "churn_project.wsgi.application"

# Database configuration (SQLite for development)
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# Localization settings
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# Static files settings
STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "churn_app/static"]  # Ensure static folder exists
STATIC_ROOT = BASE_DIR / "staticfiles"  # For collectstatic (used in production)

# Default primary key field type
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LOGIN_REDIRECT_URL = "/"
LOGOUT_REDIRECT_URL = "/login/"

# Redirect users after login/logout
LOGIN_REDIRECT_URL = "predict_churn"
LOGOUT_REDIRECT_URL = "home"
