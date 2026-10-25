from app.config import get_settings

settings = get_settings()


print(f"Environment: {settings.app_env}")
print(f"Primary_model: {settings.primary_model}")
print(f"Rate limit: {settings.rate_limit}")
print(f"Cache TTL: {settings.cache_ttl_seconds}")
print(f"Max retries: {settings.max_retries}")
print(f"Is produsction: {settings.is_production}")
print("Config loaded successfully!")
