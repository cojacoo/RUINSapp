================
Deployment guide
================

The RUINSapp consists of many different sub-applications. For this guide we assume that the full application
should be deployed on a single server instance.

Webserver
=========

We need at least one instance serving the main website. In this guide, nginx is used.
Install:

```bash
apt update && apt install -y nginx
```

Next, we need certbot to automatically install and auto-renew SSL certificates.
Recommended way to install certbot is using snap, which needs to be installed in most cases first.

```bash
apt install snapd
snap install core
snap install certbot --classic
```

Then configure certbot:

```bash
ln -s /snap/bin/certbot /usr/bin/certbot
```

Then request and install certificates for nginx

```bash
certbot --nginx
```

