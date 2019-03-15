include($${PWD}/../../common.pri)
unix:include($${PWD}/../../cuda_compiler.pri)
QT      += testlib
TARGET = DeviceBM
DEFINES += DEVICE_BM
OBJECTS_DIR=obj
unix:{
   LIBS+= -lgtest -lbenchmark
}
LIBS+= -L"../../FlockGPU" -lFlockGPU
DESTDIR=./