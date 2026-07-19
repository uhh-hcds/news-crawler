FROM python:3.13

WORKDIR /app

# cron drives the schedule (hourly feed polling, nightly corpus export), the
# same way the Go image ran crond. Without it the container scrapes once and
# exits, and `restart: on-failure` never brings it back after a clean exit.
RUN apt-get update \
 && apt-get install -y --no-install-recommends cron \
 && rm -rf /var/lib/apt/lists/*

# pip but faster
RUN pip install uv

COPY app/requirements.txt requirements.txt

RUN uv pip install --system --no-cache-dir -r requirements.txt

# Bake the Punkt sentence models in. Downloading them at runtime would make
# every cron run depend on nltk.org being reachable, and a silent fallback to
# the untrained tokenizer would quietly shred German abbreviations.
ENV NLTK_DATA=/usr/share/nltk_data
RUN python -m nltk.downloader -d ${NLTK_DATA} punkt punkt_tab \
 && python -c "import nltk; nltk.data.load('tokenizers/punkt/german.pickle'); print('punkt german OK')"

COPY app .
COPY crontab /app/crontab
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

RUN chmod +x /usr/local/bin/docker-entrypoint.sh && crontab /app/crontab

CMD ["/usr/local/bin/docker-entrypoint.sh"]
