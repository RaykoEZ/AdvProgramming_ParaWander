include($${PWD}/../../common.pri)

QT      += testlib
TARGET = DeviceBM
DEFINES += DEVICE_BM
OBJECTS_DIR=obj
CUDA_OBJECTS_DIR = cudaobj
# and add the include dir into the search path for Qt and make
INCLUDEPATH +=./include \
              $$PWD/../../FlockGPU/include 

HEADERS += \
    $$PWD/src/*.h \
    $$PWD/src/*.cuh

SOURCES += $$PWD/src/main.cpp

unix:{
   LIBS+= -lbenchmark
}
# link with my libraries
LIBS+= -L"../../FlockCPU" -lFlockCPU
LIBS+= -L"../../FlockGPU" -lFlockGPU
QMAKE_RPATHDIR += ../../FlockGPU
QMAKE_RPATHDIR += ../../FlockCPU
# where our exe is going to live (root of project)
DESTDIR=./


CUDA_SOURCES += $$files($$PWD/src/*.cu)


unix:include($${PWD}/../../cuda_compiler.pri)



