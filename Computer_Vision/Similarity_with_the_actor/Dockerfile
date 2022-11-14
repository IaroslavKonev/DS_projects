# install the face_recognition image
FROM animcogn/face_recognition:latest
# You can use different images here, for a simpler python project
# FROM python:3.10.2-slim-buster

# copy the file with the necessary libraries that we want to install
COPY ./requirements.txt /root/requirements.txt

# update pip and install the libraries from the requirements, --ignore-installed - reinstall packages if they already exist
RUN pip install --upgrade pip && \
    pip install --ignore-installed -r /root/requirements.txt

# creating a working directory
WORKDIR /root/docker_test

# copying all files that are not specified in dockerignore to a new directory
COPY . /root/docker_test

# running the script, for the new project put train.py and make key_load_img: True in the params.yaml file
CMD ["python", "inference.py"]
