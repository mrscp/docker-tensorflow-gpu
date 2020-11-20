# Docker TensorFlow GPU [NVIDIA]

# Instruction

* Build and start development server
    ```shell script
    docker-compose up --build -d
    ```

* Execute within the container
    ```shell script
    docker exec -it docker_tf_gpu bash 
    ```
  
* Running python scripts
    ```shell script
    python main.py
    ```

Documentation about NVIDIA settings for docker: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html