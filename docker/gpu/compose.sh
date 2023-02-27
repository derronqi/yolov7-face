export USERID=$(id -u)
export GROUPID=$(id -g)
export USERNAME=$(whoami)
export CONTAINERNAME="yolo_2210_pdh"
printf "USERID=%s\n" $USERID 
printf "GROUPID=%s\n" $GROUPID 
printf "USERNAME=%s\n" $USERNAME 
xhost +
xhost +local:docker
echo $xhost
#docker-compose up -d --force-recreate --no-deps --build
docker rm -f ${CONTAINERNAME}
docker-compose up -d --build  --force-recreate
docker exec ${CONTAINERNAME} echo "xhost's display : "
docker exec ${CONTAINERNAME} echo $DISPLAY
docker exec ${CONTAINERNAME} echo "visualize test"
docker exec ${CONTAINERNAME} xclock
docker exec -it ${CONTAINERNAME} /bin/bash
