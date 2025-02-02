import os

workers = int(os.getenv("GUNICORN_WORKERS", "2"))
worker_class = "uvicorn.workers.UvicornWorker"
bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"
keepalive = 120
errorlog = "-"
accesslog = "-"
worker_tmp_dir = "/dev/shm"