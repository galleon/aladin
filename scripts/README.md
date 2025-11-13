# Scripts

This directory contains utility scripts for managing the Aladin platform.

## recreate-kind-cluster.sh

Recreates the kind management cluster with proper Docker socket access for Cluster API Provider Docker (CAPD).

### Usage

```bash
./scripts/recreate-kind-cluster.sh
```

### What it does

1. **Checks prerequisites**: Verifies that `kind` and `clusterctl` are installed
2. **Deletes existing cluster**: Removes the existing `mgmt` kind cluster if it exists
3. **Creates new cluster**: Creates a new kind cluster with Docker socket mounted (using `kind-config.yaml`)
4. **Initializes CAPI**: Installs Cluster API providers (CAPI core + CAPD)
5. **Waits for readiness**: Ensures all controllers are ready before completing

### Requirements

- `kind` installed and in PATH
- `clusterctl` installed and in PATH
- Docker daemon running
- `kind-config.yaml` in the project root

### Why this is needed

The CAPD provider needs access to the host's Docker daemon to create containers for tenant clusters. Without mounting `/var/run/docker.sock` into the kind node, CAPD cannot provision tenant clusters and you'll see errors like:

```
Cannot connect to the Docker daemon at unix:///var/run/docker.sock
```

### After running

After successfully running this script:

1. Restart your Docker Compose application (if running)
2. Existing tenants may need to be recreated (their clusters won't work without Docker access)
3. New tenants should provision successfully

