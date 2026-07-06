## Setting up Docker and executing DVC pipeline using docker

```bash
# 1. Build the Docker Image
docker-compose build
# 2. Run the Container
docker-compose up
```

```bash
# 3. Stop and Remove Containers
docker-compose down
```

## Additional Useful docker commands

```bash
# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# List Docker images
docker images
```

```bash
# Remove all stopped containers
docker container prune

# Remove specific containers (replace <container_id_or_name>)
docker rm <container_id_or_name>
```