from django.apps import AppConfig
from .views import init_schema

class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'

    def ready(self):
        try:
            init_schema()
            print(" Weaviate schema initialized successfully")
        except Exception as e:
            print(f" Schema initialization failed: {e}") 