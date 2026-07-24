# Extends the redhat-actions runner image, replacing only the outdated runner
# binary (2.289.2 → current). All existing entrypoint/register scripts are kept.
#
# quay.io/redhat-github-actions/runner:latest has not been updated since ~2022
# and GitHub's backend rejects the old runner version with 401.
ARG RUNNER_VERSION=2.335.1

# Stage 0: source of OpenSSL 3 libs (UBI9 ships libssl.so.3 / libcrypto.so.3)
FROM registry.access.redhat.com/ubi9/ubi-minimal:latest AS openssl3

FROM quay.io/redhat-github-actions/runner:latest

ARG RUNNER_VERSION
USER root

# Base image is Fedora 35 with OpenSSL 1.1.x. bitwarden/sm-action@v2 requires
# the actual OpenSSL 3 versioned symbol OPENSSL_3.0.0. Symlinks to 1.1.x don't
# satisfy versioned symbol requirements. Copy real OpenSSL 3 libs from UBI9.
RUN --mount=from=openssl3,source=/usr/lib64,target=/ubi9-lib64 \
    cp $(ls /ubi9-lib64/libssl.so.3* /ubi9-lib64/libcrypto.so.3* 2>/dev/null | tr '\n' ' ') /usr/lib64/ && \
    ldconfig

RUN curl -fsSL \
    https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz \
    -o /tmp/runner.tar.gz && \
    tar xzf /tmp/runner.tar.gz -C /home/runner --overwrite && \
    rm /tmp/runner.tar.gz && \
    chmod -R 777 /home/runner

# Pre-install oc CLI so the deploy job doesn't need to download it at runtime.
RUN curl -fsSL https://mirror.openshift.com/pub/openshift-v4/clients/ocp/stable/openshift-client-linux.tar.gz \
      -o /tmp/oc.tar.gz && \
    tar xzf /tmp/oc.tar.gz -C /usr/local/bin oc kubectl && \
    rm /tmp/oc.tar.gz && \
    oc version --client

USER 1001
