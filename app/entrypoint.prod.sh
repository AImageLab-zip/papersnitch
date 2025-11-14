#!/usr/bin/env bash

#python manage.py collectstatic --noinput
#python manage.py migrate --noinput
uv run --host 0.0.0.0 app
python -m gunicorn --bind 0.0.0.0:8000 --workers 3 web.wsgi:application