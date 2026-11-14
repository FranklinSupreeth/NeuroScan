# Gunicorn settings for Render.
# One worker keeps memory under Render's free 512 MB limit
# (the SVM model alone uses ~400 MB once loaded).
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
workers = int(os.environ.get("WEB_CONCURRENCY", 1))
threads = int(os.environ.get("GUNICORN_THREADS", 2))
timeout = 120
preload_app = True
accesslog = "-"
errorlog = "-"
