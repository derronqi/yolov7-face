PROJECT_HOME = .
OPENCV_PATH= ${PROJECT_HOME}/libs/opencv
CUXX = g++

BIN = ${PROJECT_HOME}/bin/yoloface

SRCS = ./src/yolov7face.cpp ./src/main.cpp
OBJS = $(SRCS:.cpp=.o)

CFLAGS = -std=c++11 -fpermissive -O3 -Iinclude -I${PROJECT_HOME}/include/ncnn 
LIBFLAGS = -fopenmp -L${OPENCV_PATH} -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio -lopencv_highgui -L${PROJECT_HOME}/libs -lncnn

all: $(BIN)

$(OBJS):

%.o: %.cpp
	$(CUXX) -c $(CFLAGS) $< -o $@

$(BIN): $(OBJS)
	$(CUXX) $^ $(LIBFLAGS) -o $@ 

clean:
	rm -f $(BIN) ./src/*.o ./bin/*.jpg 
