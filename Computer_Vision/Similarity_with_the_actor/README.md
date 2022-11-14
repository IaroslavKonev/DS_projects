# Tutorial docker for Data Science project

- Installing Docker
https://docs.docker.com/get-docker/

- Assemble the image
```
docker build -t your-name-image .
```

- View all collected images
```
docker images
```

- Delete Docker image
```
docker rmi your-id-image
```

- Build an application from a Docker image (container)
```
docker run your-name-image
```

- If we want to run a particular, for example, script inside the image
```
docker run your-name-image python train.py
```

- View all running containers
```
docker ps   
```

- View all running/non-running containers
```
docker ps -a
```

- Stop a running specific container
```
docker stop my_container
```

- Stop all running containers
```
docker stop $(docker ps -a -q)
```

- Remove all containers (if any)
```
docker container rm $(docker ps -a -q)
```
