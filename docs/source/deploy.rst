================
Deployment guide
================

The RUINSapp consists of many different sub-applications. For this guide we assume that the full application
should be deployed on a single server instance.

Webserver
=========

Install
-------

We need at least one instance serving the main website. In this guide, nginx is used.
Install:

.. code-block:: bash

    apt update && apt install -y nginx


Next, we need certbot to automatically install and auto-renew SSL certificates.
Recommended way to install certbot is using snap, which needs to be installed in most cases first.

.. code-block:: bash

    apt install snapd
    snap install core
    snap install certbot --classic


Then configure certbot:

.. code-block:: bash
    ln -s /snap/bin/certbot /usr/bin/certbot


Then request and install certificates for nginx

.. code-block:: bash
    certbot --nginx

Configure website
-----------------

.. todo::
    Write this

Docker
======

Install docker on the host machine:

.. code-block:: bash
    
    apt install -y ca-certificates curl gnupg lsb-release

Add dockers GPG key and add package repository to sources

.. code-block:: bash

    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

    echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

Install docker

.. code-bock:: bash

    apt update
    apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin


aaaand finally:

```bash
docker run -d -i  -p 42001:8501  --name weather  --restart always  ghcr.io/hydrocode-de/ruins:v0.6.0 weather.py
```

