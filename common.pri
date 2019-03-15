# as I want to support 4.8 and 5 this will set a flag for some of the mac stuff
# mainly in the types.h file for the setMacVisual which is native in Qt5
QT       += core
isEqual(QT_MAJOR_VERSION, 5) {
        cache()
        DEFINES +=QT5BUILD
}
# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS \
           GLM_ENABLE_EXPERIMENTAL

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0
# on a mac we don't create a .app bundle file ( for ease of multiplatform use)
CONFIG-=app_bundle
#CONFIG += console c++11
# and add the include dir into the search path for Qt and make

unix {
    target.path = /usr/lib
    INSTALLS += target
}
win32:
{
    # GLM AT HOME
    INCLUDEPATH+= $$PWD/../../glm

}

CONFIG-=app_bundle
# flags taken from Jack Diver's qt setup:
# https://github.com/nitronoid/flo/blob/master/common.pri
unix:{
QMAKE_CXXFLAGS += -std=c++11 -g -fdiagnostics-color
# Optimisation flags
QMAKE_CXXFLAGS += -Ofast -march=native -frename-registers -funroll-loops -ffast-math -fassociative-math

# Intrinsics flags
QMAKE_CXXFLAGS += -mfma -mavx2 -m64 -msse -msse2 -msse3
# Enable all warnings
QMAKE_CXXFLAGS += -Wall -Wextra -pedantic-errors -Wno-sign-compare
# Vectorization info
QMAKE_CXXFLAGS += -ftree-vectorize -ftree-vectorizer-verbose=5

}
