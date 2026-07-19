#!/bin/bash
set -e

# cron runs jobs with a near-empty environment and never sees the variables
# docker passes to the container, so persist them where the crontab entries
# can source them. shlex.quote keeps passwords with shell metacharacters intact.
#
# PATH is written explicitly: the crontab also sets it, but if someone edits the
# crontab and drops that line, the jobs still find python at /usr/local/bin
# rather than failing silently with "python: not found".
python - <<'PY' > /app/crawler.env
import os, shlex
print("export PATH=/usr/local/bin:/usr/bin:/bin")
for key, value in sorted(os.environ.items()):
    if key.startswith(("CRAWLER_", "ELASTIC_")):
        print(f"export {key}={shlex.quote(value)}")
PY
chmod 0600 /app/crawler.env

mkdir -p /app/out
touch /app/out/events.log

# Elasticsearch takes a while to accept connections; without this the first
# run after `docker compose up` fails against a cluster that is still booting.
if [ -n "${ELASTIC_URL:-}" ]; then
  echo "Waiting for Elasticsearch at ${ELASTIC_URL} ..."
  python - <<'PY'
import os, sys, time, urllib.request
url = os.environ["ELASTIC_URL"]
for attempt in range(60):
    try:
        urllib.request.urlopen(url, timeout=5).read()
        print("Elasticsearch is up")
        sys.exit(0)
    except Exception:
        time.sleep(5)
print("Elasticsearch did not become ready; cron will retry on schedule", file=sys.stderr)
PY
fi

# A fresh stack otherwise sits empty until the next :30. Off by default so the
# behaviour matches the Go image, which only ever ran on its cron schedule.
if [ "${CRAWLER_RUN_ON_START:-false}" = "true" ]; then
  echo "CRAWLER_RUN_ON_START=true -> running an initial scrape now"
  ( . /app/crawler.env; cd /app && python main.py scrape --day all --no-corpus --verbose ) \
    >> /app/out/events.log 2>&1 || echo "initial scrape failed; cron will retry"
fi

echo "Starting cron. Job output -> /app/out/events.log (and this log)."
cron
exec tail -F /app/out/events.log
