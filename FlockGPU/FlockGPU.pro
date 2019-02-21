QT       += core
TARGET = FlockGPU

# where to put moc auto generated files
MOC_DIR=moc
DEFINES += FLOCKGPU_LIBRARY
OBJECTS_DIR=obj
# as I want to support 4.8 and 5 this will set a flag for some of the mac stuff
# mainly in the types.h file for the setMacVisual which is native in Qt5
isEqual(QT_MAJOR_VERSION, 5) {
        cache()
        DEFINES +=QT5BUILD
}
# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS
TEMPLATE = lib
CONFIG += staticlib


# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0
# on a mac we don't create a .app bundle file ( for ease of multiplatform use)
CONFIG-=app_bundle

# and add the include dir into the search path for Qt and make
INCLUDEPATH +=./include
# where our exe is going to live (root of project)
DESTDIR=./

SOURCES += \
    $$PWD/src/main.cpp

# cuda project setuo from Richard Southern's libfluid:
#https://github.com/NCCA/libfluid/tree/master/libfluid

LIB_INSTALL_DIR=$$PWD/lib
BIN_INSTALL_DIR=$$PWD/bin
INC_INSTALL_DIR=$$PWD/include

linux:QMAKE_CXX = $$(HOST_COMPILER)
macx:QMAKE_CXX=clang++
QMAKE_CXXFLAGS += -D_DEBUG -DTHRUST_DEBUG

INCLUDEPATH += ${CUDA_SAMPLES_PATH}/common/inc ${PWD}/../common/include include ${CUDA_PATH}/include ${CUDA_PATH}/include/cuda

# Set this up as the installation directory for our library
TARGET = $$LIB_INSTALL_DIR/flock

# Set the C++ flags for this compilation when using the host compiler
QMAKE_CXXFLAGS += -std=c++11 -fPIC
HEADERS+=INC_INSTALL_DIR/*.cuh

# Directories
INCLUDEPATH += ${CUDA_SAMPLES}/common/inc ${PWD}/../common/include
QMAKE_CXXFLAGS += $$system(pkg-config --silence-errors --cflags cuda-8.0 cudart-8.0 curand-8.0 cublas-8.0)

# Link with the following libraries
LIBS += $$system(pkg-config --silence-errors --libs cuda-8.0 cudart-8.0 curand-8.0 cublas-8.0) -lcublas_device -lcudadevrt

# Use the following path for nvcc created object files
CUDA_OBJECTS_DIR = cudaobj

# CUDA_DIR - the directory of cuda such that CUDA\<version-number\ contains the bin, lib, src and include folders
# Set this environment variable yourself.
CUDA_DIR=$$system(pkg-config --silence-errors --variable=cudaroot cuda-8.0)
isEmpty(CUDA_DIR) {
    message(CUDA_DIR not set - set this to the base directory of your local CUDA install (on the labs this should be /usr))
}

## CUDA_SOURCES - the source (generally .cu) files for nvcc. No spaces in path names
CUDA_SOURCES += src/*.cu

## CUDA_INC - all includes needed by the cuda files (such as CUDA\<version-number\include)
CUDA_INC += $$join(INCLUDEPATH,' -I','-I',' ')

# nvcc flags ("-Xptxas -v" option is always useful, "-D_DEBUG" for tons of debug info)
NVCC_DEBUG_FLAGS =
#NVCC_DEBUG_FLAGS += -D_DEBUG -g -G  -DTHRUST_DEBUG
#NVCC_DEBUG_FLAGS += -Xptxas -v
# New added by Jon - determines the best current cuda architecture of your system
GENCODE=$$system(../findCudaArch.sh)
NVCCFLAGS =  -ccbin $$(HOST_COMPILER) -I../src/	-m64 $$NVCC_DEBUG_FLAGS $$GENCODE --compiler-options -fno-strict-aliasing --compiler-options -fPIC -use_fast_math --std=c++11
#message($$NVCCFLAGS)
# Define the path and binary for nvcc
NVCCBIN = $$CUDA_DIR/bin/nvcc

#prepare intermediate cuda compiler
cuda.input = CUDA_SOURCES
cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}.o
cuda.commands = $$NVCCBIN $$NVCCFLAGS -dc $$CUDA_INC $$LIBS ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}

#Set our variable out. These obj files need to be used to create the link obj file
#and used in our final gcc compilation
cuda.variable_out = CUDA_OBJ
cuda.variable_out += OBJECTS
cuda.clean = $$CUDA_OBJECTS_DIR/*.o
# Note that cuda objects are linked separately into one obj, so these intermediate objects are not included in the final link
cuda.CONFIG = no_link
QMAKE_EXTRA_COMPILERS += cuda

# Prepare the linking compiler step (combine tells us that the compiler will combine all the input files)
cudalink.input = CUDA_OBJ
cudalink.CONFIG = combine
cudalink.output = $$OBJECTS_DIR/cuda_link.o

# Tweak arch according to your hw's compute capability
cudalink.commands = $$NVCCBIN $$NVCCFLAGS $$CUDA_INC -dlink ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} $$LIBS
cudalink.dependency_type = TYPE_C
cudalink.depend_command = $$NVCCBIN $$NVCCFLAGS -M $$CUDA_INC ${QMAKE_FILE_NAME}

# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cudalink

# Set up the post install script to copy the headers into the appropriate directory
includeinstall.commands = mkdir -p $$INC_INSTALL_DIR && cp include/*.h $$INC_INSTALL_DIR
QMAKE_EXTRA_TARGETS += includeinstall
POST_TARGETDEPS += includeinstall
