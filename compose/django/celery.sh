#!/bin/sh

LOGFILE="$LOGS_PATH/celery.log"

# wait for migrations
sleep 10

exec /usr/local/bin/celery worker \
    --app=genomics_network \
    --loglevel=INFO \
    --logfile=$LOGFILE \
    --soft-time-limit=90 \
    --time-limit=120 \
    --uid django
