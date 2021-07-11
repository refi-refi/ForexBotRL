# Get a Debian image with Python 3.6.8
FROM python:3.6.8

# Set Bash as the default shell
SHELL ["/bin/bash", "-c"]

# Copy project to docker image
WORKDIR ForexBot
COPY ForexBot ForexBot

# Update system and install some utilities
RUN apt-get update && \
				apt-get upgrade -y && \
				apt-get install -y cmake libopenmpi-dev python3-dev zlib1g-dev ffmpeg libsm6 libxext6 vim htop && \
				apt-get clean

# Set-up python environment
ENV VIRTUAL_ENV=/ForexBot/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies
RUN pip install --upgrade pip && \
				pip install wheel && \
				pip install -e ForexBot

# Launch optimization
# CMD ["python", "ForexBot/forex_bot/main/optimize/a2c_optimize/a2c_optuna.py"]

