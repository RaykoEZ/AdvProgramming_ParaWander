include($${PWD}/../common.pri)


TARGET = FlockGPU
DEFINES += FLOCKGPU_LIBRARY
OBJECTS_DIR=obj

TEMPLATE = lib
CONFIG += staticlib
# where our exe is going to live (root of project)
DESTDIR=./

SOURCES += \
    $$PWD/src/*.cpp \
    src/*.cu \


HEADERS += \
    $$PWD/include/*.cuh \
    $$PWD/include/*.h \


LIB_INSTALL_DIR=$$PWD/lib
BIN_INSTALL_DIR=$$PWD/bin
INC_INSTALL_DIR=$$PWD/include
#
linux:QMAKE_CXX = $$(HOST_COMPILER)
macx:QMAKE_CXX=clang++
QMAKE_CXXFLAGS += -D_DEBUG -DTHRUST_DEBUG
#
INCLUDEPATH += ${CUDA_SAMPLES}/common/inc ${PWD}/../common/include include ${CUDA_PATH}/include ${CUDA_PATH}/include/cuda 
# and add the include dir into the search path for Qt and make
INCLUDEPATH +=./include
#
# # Set this up as the installation directory for our library
TARGET = $$LIB_INSTALL_DIR/flock


CUDA_OBJECTS_DIR = cudaobj
CUDA_SOURCES += $$files($$PWD/src/*.cu)


include($${PWD}/../cuda_compiler.pri)