include($${PWD}/../common.pri)

QT      += testlib
TARGET = FlockTest
DEFINES += FLOCK_TEST
OBJECTS_DIR=obj

# and add the include dir into the search path for Qt and make
INCLUDEPATH +=./include \
              $$PWD/../FlockGPU/include \
              $$PWD/../FlockCPU/include \

HEADERS += \
    src/UnitTests.h \

SOURCES += $$PWD/src/main.cpp

unix:{
   LIBS+= -lgtest -lbenchmark
}
# link with my libraries
LIBS+= -L"../FlockCPU" -lFlockCPU
LIBS+= -L"../FlockGPU" -lFlockGPU
# where our exe is going to live (root of project)
DESTDIR=./
win32:{

    # Gtest dependencies at Home, according to https://doc.qt.io/qtcreator/creator-autotest.html#setting-up-the-google-c-testing-framework
    GTEST_DIR = ../../googletest
    INCLUDEPATH+= $$GTEST_DIR/googletest \
                  $$GTEST_DIR/googletest/include \
                  $$GTEST_DIR/googlemock \
                  $$GTEST_DIR/googlemock/include
    # sources needed for gtest
    SOURCES += $$GTEST_DIR/googletest/src/gtest-all.cc \
               $$GTEST_DIR/googlemock/src/gmock-all.cc
}

unix:include($${PWD}/../cuda_compiler.pri)

