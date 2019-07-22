include($${PWD}/../common.pri)

QT      += testlib
TARGET = FlockDemo
DEFINES += FLOCK_DEMO
OBJECTS_DIR=obj
CUDA_OBJECTS_DIR = cudaobj
# and add the include dir into the search path for Qt and make
INCLUDEPATH +=./include \
              $$PWD/../FlockGPU/include \
              $$PWD/../FlockCPU/include \

HEADERS += \
    $$PWD/src/*.h

SOURCES += $$PWD/src/*.cpp

# unix:{
#   LIBS+= -lgtest -lbenchmark
# }
# link with my libraries
LIBS+= -L"../FlockCPU" -lFlockCPU
LIBS+= -L"../FlockGPU" -lFlockGPU
QMAKE_RPATHDIR += ../FlockGPU
QMAKE_RPATHDIR += ../FlockCPU
# where our exe is going to live (root of project)
DESTDIR=./



CUDA_SOURCES += $$files($$PWD/src/*.cu)


unix:include($${PWD}/../cuda_compiler.pri)



