include($${PWD}/../common.pri)
TARGET = FlockCPU
DEFINES += FLOCKCPU_LIBRARY
OBJECTS_DIR=obj


TEMPLATE = lib
CONFIG += staticlib

# and add the include dir into the search path for Qt and make
INCLUDEPATH +=./include
# where our exe is going to live (root of project)
DESTDIR=./

SOURCES += \
    src/World.cpp \
    src/Boid.cpp \
    src/FlockActions.cpp \

HEADERS += \
    include/World.h \
    include/Boid.h \
    include/FlockActions.h \






