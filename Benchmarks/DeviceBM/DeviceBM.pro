include($${PWD}/../../common.pri)
CONFIG += console
TEMPLATE = app
TARGET = Flock_DeviceBM.out
OBJECTS_DIR = obj
CUDA_OBJECTS_DIR = cudaobj
# and add the include dir into the search path for Qt and make
INCLUDEPATH +=./include \
              $$PWD/../../FlockGPU/include \
              $$PWD/../include

HEADERS += \
    $$PWD/../../FlockGPU/include/*.h \
    $$PWD/../../FlockGPU/include/*.cuh \
    $$PWD/src/*.cuh \
    $$PWD/../include/BenchCommon.h

SOURCES += ./src/main.cpp
unix:{
   LIBS+= -lbenchmark
}

# link with my libraries
LIBS+= -L"../../FlockGPU" -lFlockGPU
QMAKE_RPATHDIR += ../../FlockGPU

CUDA_SOURCES += $$files($$PWD/src/*.cu)


unix:include($${PWD}/../../cuda_compiler.pri)



