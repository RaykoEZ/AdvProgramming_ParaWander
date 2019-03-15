include($${PWD}/../../common.pri)

QT      += testlib
TARGET = HostBM
DEFINES += HOST_BM
OBJECTS_DIR=obj
unix:{
   LIBS+= -lgtest -lbenchmark
}
LIBS+= -L"../../FlockCPU" -lFlockCPU
DESTDIR=./