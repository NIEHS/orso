# ORSO Developer Manual

## ORSO deployment

ORSO is implemented in the [Django web framework](https://www.djangoproject.com/). It is designed to be rapidly deployed using [Docker Compose](https://docs.docker.com/). Provided that Docker and Docker Compose are available, ORSO may be deployed using a single command. For instructions on installation of Docker, see the documentation for [Docker](https://docs.docker.com/install/) and [Docker Compose](https://docs.docker.com/compose/).

To deploy, use the following commands to clone the ORSO repository and run the Docker image:

```
git clone https://github.com/lavenderca/genomics-network
cd genomics-network
docker-compose up
```
