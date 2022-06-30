FROM debian:unstable-20220622-slim

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
        && apt-get -y install --no-install-recommends \
        wget=1.21.3-1+b2 \
        shellcheck=0.8.0-2 \
        ca-certificates=20211016 \
        python3-pip=22.1.1+dfsg-1 \
        python3-pytest=7.1.2-2 \
        mypy=0.961-1 \
        && rm -rf /var/lib/apt/lists

RUN set -o pipefail; wget --progress=dot:giga -O /bin/hadolint https://github.com/hadolint/hadolint/releases/download/v2.10.0/hadolint-Linux-x86_64 \
        && sha256sum /bin/hadolint \
        && echo 8ee6ff537341681f9e91bae2d5da451b15c575691e33980893732d866d3cefc4 /bin/hadolint | sha256sum -c \
        && chmod a+x /bin/hadolint

COPY requirements.txt /
RUN pip3 install --no-cache-dir -r /requirements.txt
