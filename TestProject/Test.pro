QT      += core testlib
TARGET = FlockTest
# TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
# where to put moc auto generated files
MOC_DIR = moc
DEFINES += FLOCK_TEST
OBJECTS_DIR=obj

# and add the include dir into the search path for Qt and make
INCLUDEPATH +=./include \
              $$PWD/../FlockGPU/include \
              $$PWD/../FlockCPU/include \


SOURCES += $$PWD/src/main.cpp
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
unix:{
   LIBS+= -lgtest -lbenchmark
}
DEFINES += QT_DEPRECATED_WARNINGS
CONFIG-=app_bundle

LIBS+= -L"../FlockCPU" -lFlockCPU
# where our exe is going to live (root of project)
DESTDIR=./
win32:{

    # GLM AT HOME
    INCLUDEPATH+= $$PWD/../../../glm

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

HEADERS += \
    src/UnitTests.h \

