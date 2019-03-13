QT       += core
TARGET = FlockCPU

# where to put moc auto generated files
MOC_DIR=moc
DEFINES += FLOCKCPU_LIBRARY
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
DEFINES += QT_DEPRECATED_WARNINGS \
           GLM_ENABLE_EXPERIMENTAL
TEMPLATE = lib
CONFIG += staticlib


# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0
# on a mac we don't create a .app bundle file ( for ease of multiplatform use)
CONFIG-=app_bundle
#CONFIG += c++11
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

unix {
    target.path = /usr/lib
    INSTALLS += target
}
CONFIG += console
win32:
{
    # GLM AT HOME
    INCLUDEPATH+= $$PWD/../../../glm

}


