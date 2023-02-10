export USERID=$(id -u)
export GROUPID=$(id -g)
export USERNAME=$(whoami)
export CONTAINERNAME="yolo_jetson_pdh_rs"
printf "USERID=%s\n" $USERID 
printf "GROUPID=%s\n" $GROUPID 
printf "USERNAME=%s\n" $USERNAME 
xhost +
xhost +local:docker
echo $xhost
docker rm -f ${CONTAINERNAME}
#docker-compose up -d 
#docker-compose up -d --force-recreate --no-deps --build --no-cache
#docker-compose rm -f
docker-compose build #--no-cache
docker-compose up -d

docker exec ${CONTAINERNAME} echo "xhost's display : "
docker exec ${CONTAINERNAME} echo $DISPLAY
docker exec ${CONTAINERNAME} echo "visualize test"
docker exec ${CONTAINERNAME} xclock
#docker-compose exec echo ptgaze_jetson_image "xhost : " $xhost
#docker-compose exec echo ptgaze_jetson_image "visualize tool test (xclock)"
docker exec -it ${CONTAINERNAME} /bin/bash
