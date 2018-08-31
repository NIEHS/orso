# ORSO Developer Manual

## ORSO deployment

ORSO is implemented in the [Django web framework](https://www.djangoproject.com/). It is designed to be rapidly deployed using [Docker Compose](https://docs.docker.com/). Provided that Docker and Docker Compose are available, ORSO may be deployed using a single command. For instructions on installation of Docker, see the documentation for [Docker](https://docs.docker.com/install/) and [Docker Compose](https://docs.docker.com/compose/).

To deploy, first clone the ORSO codebase to your machine and navigate to the cloned directory using the following command:

```
git clone https://github.com/lavenderca/genomics-network
cd genomics-network
```

You will likely need to [tailor the nginx configuration file to your specifications](http://nginx.org/en/docs/http/configuring_https_servers.html). The nginx configuration file used in deployment can be found here:

```
compose/nginx/conf/nginx.conf
```

ORSO is configured to read critical variables, such as usernames and keys, from the environmental variable file `.env`. We include a sample file stripped of sensitive usernames and keys at `env.example`. Before deploying ORSO, fill out `env.example`, and copy to `.env`.

After correctly setting up your nginx configuration and `.env` files,  deploy using the following Docker Compose command:

```
docker-compose up
```

## Database population

Docker Compose will correctly set up tables in the PostgreSQL database, but the database will need to be populated with biologically relevant features like genes and enhancers. Population of the database is controlled by custom Django management commands. To use these commands within the Docker instance, the `docker-compose exec` function will be used. The following command performs the initial population of the database:

```
docker-compose exec --user django -d django sh -c \
    "python manage.py initial_setup > initial_setup.log 2>&1"
```

This command performs the following operations:

* Creation of organism objects (human, mouse, fly, and worm).
* Creation of assembly objects (GRCh38, hg19, mm10, mm9, dm6, dm3, ce11, and ce10).
* Query of RefSeq and creation of associated gene objects.
* Creation of enhancer objects. Human and mouse enhancers are taken from the from the [VISTA database](https://enhancer.lbl.gov/). Fly and worm enhancer loci are taken from [Kvon *et al.*](http://enhancers.starklab.org/) and [Chen *et al.*](https://www.ncbi.nlm.nih.gov/pubmed/23550086), respectively.

ORSO should now function properly, allowing for user accounts and datasets to be added using its web interface. However, many ORSO features, like its similarity and recommendation systems, are only useful if many datasets are hosted. Because of this, we recommend first adding data and then running commands to update ORSO's machine learning models. To facilitate this, we include an automated command to add ENCODE data to ORSO.

## Adding data from ENCODE

ORSO provides an function to automatically query the [ENCODE](https://www.encodeproject.org/) API and retrieve all available coverage files. This can be used to populate your ORSO instance with validated data from a large consortial dataset. The command is as follows:

```
docker-compose exec --user django -d django sh -c \
    "python manage.py update_encode > update_encode.log 2>&1"
```

Use this command with caution. It is computationally demanding and requires significant bandwidth and memory to run efficiently. Even on a big server, expect this command to take 4 or 5 days to reach completion.

## Updating models

ORSO runs underlying PCA and MLP models for evaluating dataset similarity and for displaying data.  The following command will update all models in the ORSO instance:

```
docker-compose exec --user django -d django sh -c \
    "python manage.py update_models \
        --transcript_selections \
        --feature_attributes \
        --feature_values \
        --pcas \
        --mlps \
        --predictions \
        --similarities \
        --recommendations \
        --dendrograms \
        --networks \
    > update_models.log 2>&1"
```
