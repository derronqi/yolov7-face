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
docker-compose rm -f 
docker-compose up -d --build 
docker-compose exec echo $xhost
docker exec -it ${CONTAINERNAME} /bin/bash
