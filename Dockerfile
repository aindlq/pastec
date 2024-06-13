# Use Ubuntu 18.04 as the base image
FROM ubuntu:18.04

# Maintainer Info
LABEL maintainer="lklic"

# Set the timezone environment variable
ENV TZ=Europe/Rome

# Set up timezone and install necessary packages
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get update && \
    apt-get install -y curl wget vim libcurl4-openssl-dev libopencv-dev libmicrohttpd-dev libjsoncpp-dev cmake build-essential

# Copy the local Pastec files to the container
COPY . /pastec

# Create necessary directories
RUN mkdir -p /pastec/build /pastec/data

# Set the working directory to the build directory
WORKDIR /pastec/build

# Run cmake and make to build Pastec
RUN cmake ../ && make

# Copy the visualWordsORB.dat file into the data directory
RUN cp /pastec/visualWordsORB.dat /pastec/data

# Expose port 4212 for Pastec server
EXPOSE 4212

# Volume to store Pastec data and index files
VOLUME /pastec/

# Command to run Pastec server
CMD ["./pastec", "-p", "4212", "/pastec/data/visualWordsORB.dat"]