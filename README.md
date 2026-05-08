# Euro-Par26 Artifact
This branch contains the source code for the creation of the artifact of 
*"Assessing the Performance Impact of Data Layouts: a Benchmarking Approach"*

The artifact is created by
1. Building the docker image with ` docker build -t artifact-image:latest .`
2. Saving the docker image using `docker save artifact-image:latest | gzip > artifact/artifact-image.tar.gz`
3. Create an archive of /artifact