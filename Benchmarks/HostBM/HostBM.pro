include($${PWD}/../../common.pri)

TEMPLATE = app
TARGET = Flock_HostBM.out
OBJECTS_DIR=obj
# and add the include dir into the search path for Qt and make
INCLUDEPATH +=./include \
              $$PWD/../../FlockCPU/include \
              $$PWD/../include

HEADERS += \
    $$PWD/../../FlockCPU/include/*.h \
    $$PWD/../include/*.h

SOURCES += $$PWD/src/*.cpp

unix:{
   LIBS+= -lbenchmark
}
# link with my libraries
LIBS+= -L"../../FlockCPU" -lFlockCPU
QMAKE_RPATHDIR += ../../FlockCPU
# where our exe is going to live (root of project)
DESTDIR=./

