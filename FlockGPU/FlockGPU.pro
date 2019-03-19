include($${PWD}/../common.pri)
TEMPLATE = lib
TARGET = FlockGPU
# where our exe is going to live (root of project)
DESTDIR=./
#DEFINES += FLOCKGPU_LIBRARY
OBJECTS_DIR = obj
CUDA_OBJECTS_DIR = cudaobj


# and add the include dir into the search path for Qt and make
INCLUDEPATH +=./include

#CONFIG += staticlib



HEADERS += \
    $$PWD/include/*.cuh \
    $$PWD/include/*.h


CUDA_SOURCES += $$files($$PWD/src/*.cu)


include($${PWD}/../cuda_compiler.pri)
