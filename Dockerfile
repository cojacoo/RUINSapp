# default to python 3.10
ARG PYTHON_VERSION=3.10

# specify a version
FROM python:${PYTHON_VERSION}
LABEL maintainer="Mirko Mälicke"

# build the structure
RUN mkdir -p /src/ruins
RUN mkdir -p /src/data

# copy the sources
COPY ./ruins /src/ruins
COPY ./data /src/data
COPY ./requirements.txt /src/requirements.txt
COPY ./setup.py /src/setup.py
COPY ./README.md /src/README.md
COPY ./LICENSE /src/LICENSE

# install
RUN pip install --upgrade pip
RUN cd /src && pip install -e .

# create the streamlit entrypoint
WORKDIR /src/ruins/apps
ENTRYPOINT ["streamlit", "run" ]
CMD [ "weather.py" ]