#!/bin/sh

python manage.py migrate --noinput
python manage.py clear_cache
python manage.py collectstatic --noinput

LOGFILE="$LOGS_PATH/gunicorn.log"

# serve w/ gunicorn
/usr/local/bin/gunicorn genomics_network.wsgi:application \
    --bind 127.0.0.1:8000 \
    --chdir=/network \
    --timeout 300 \
    --workers 3 \
    --log-level info \
    --log-file $LOGFILE \
    --max-requests 750 \
    --max-requests-jitter 250
