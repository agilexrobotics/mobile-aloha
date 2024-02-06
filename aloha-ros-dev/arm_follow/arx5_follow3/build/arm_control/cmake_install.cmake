# Install script for directory: /home/lin/aloha-ros-dev/arm_follow/arx5_follow3/src/arm_control

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/lin/aloha-ros-dev/arm_follow/arx5_follow3/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/arm_control/msg" TYPE FILE FILES
    "/home/lin/aloha-ros-dev/arm_follow/arx5_follow3/src/arm_control/msg/arx5.msg"
    "/home/lin/aloha-ros-dev/arm_follow/arx5_follow3/src/arm_control/msg/JointControl.msg"
    "/home/lin/aloha-ros-dev/arm_follow/arx5_follow3/src/arm_control/msg/JointInformation.msg"
    "/home/lin/aloha-ros-dev/arm_follow/arx5_follow3/src/arm_control/msg/ChassisCtrl.msg"
    "/home/lin/aloha-ros-dev/arm_follow/arx5_follow3/src/arm_control/msg/MagicCmd.msg"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/arm_control/cmake" TYPE FILE FILES "/home/lin/aloha-ros-dev/arm_follow/arx5_follow3/build/arm_control/catkin_generated/installspace/arm_control-msg-paths.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/lin/aloha-ros-dev/arm_follow/arx5_follow3/devel/include/arm_control")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/roseus/ros" TYPE DIRECTORY FILES "/home/lin/aloha-ros-dev/arm_follow/arx5_follow3/devel/share/roseus/ros/arm_control")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/common-lisp/ros" TYPE DIRECTORY FILES "/home/lin/aloha-ros-dev/arm_follow/arx5_follow3/devel/share/common-lisp/ros/arm_control")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gennodejs/ros" TYPE DIRECTORY FILES "/home/lin/aloha-ros-dev/arm_follow/arx5_follow3/devel/share/gennodejs/ros/arm_control")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(COMMAND "/home/lin/software/miniconda3/bin/python3" -m compileall "/home/lin/aloha-ros-dev/arm_follow/arx5_follow3/devel/lib/python3/dist-packages/arm_control")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3/dist-packages" TYPE DIRECTORY FILES "/home/lin/aloha-ros-dev/arm_follow/arx5_follow3/devel/lib/python3/dist-packages/arm_control")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/arm_control" TYPE FILE FILES "/home/lin/aloha-ros-dev/arm_follow/arx5_follow3/devel/include/arm_control/reconfigConfig.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3/dist-packages/arm_control" TYPE FILE FILES "/home/lin/aloha-ros-dev/arm_follow/arx5_follow3/devel/lib/python3/dist-packages/arm_control/__init__.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(COMMAND "/home/lin/software/miniconda3/bin/python3" -m compileall "/home/lin/aloha-ros-dev/arm_follow/arx5_follow3/devel/lib/python3/dist-packages/arm_control/cfg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3/dist-packages/arm_control" TYPE DIRECTORY FILES "/home/lin/aloha-ros-dev/arm_follow/arx5_follow3/devel/lib/python3/dist-packages/arm_control/cfg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/lin/aloha-ros-dev/arm_follow/arx5_follow3/build/arm_control/catkin_generated/installspace/arm_control.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/arm_control/cmake" TYPE FILE FILES "/home/lin/aloha-ros-dev/arm_follow/arx5_follow3/build/arm_control/catkin_generated/installspace/arm_control-msg-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/arm_control/cmake" TYPE FILE FILES
    "/home/lin/aloha-ros-dev/arm_follow/arx5_follow3/build/arm_control/catkin_generated/installspace/arm_controlConfig.cmake"
    "/home/lin/aloha-ros-dev/arm_follow/arx5_follow3/build/arm_control/catkin_generated/installspace/arm_controlConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/arm_control" TYPE FILE FILES "/home/lin/aloha-ros-dev/arm_follow/arx5_follow3/src/arm_control/package.xml")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/arm_control/arm_3" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/arm_control/arm_3")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/arm_control/arm_3"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/arm_control" TYPE EXECUTABLE FILES "/home/lin/aloha-ros-dev/arm_follow/arx5_follow3/devel/lib/arm_control/arm_3")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/arm_control/arm_3" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/arm_control/arm_3")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/arm_control/arm_3"
         OLD_RPATH "/opt/ros/noetic/lib:/home/lin/aloha-ros-dev/arm_follow/arx5_follow3/devel/lib:/home/lin/aloha-ros-dev/arm_follow/arx5_follow3/src/arm_control/src/App:/home/lin/aloha-ros-dev/arm_follow/arx5_follow3/src/arm_control/src/Hardware:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/arm_control/arm_3")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/arm_3" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/arm_3")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/arm_3"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/lin/aloha-ros-dev/arm_follow/arx5_follow3/devel/lib/arm_control/arm_3")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/arm_3" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/arm_3")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/arm_3"
         OLD_RPATH "/opt/ros/noetic/lib:/home/lin/aloha-ros-dev/arm_follow/arx5_follow3/devel/lib:/home/lin/aloha-ros-dev/arm_follow/arx5_follow3/src/arm_control/src/App:/home/lin/aloha-ros-dev/arm_follow/arx5_follow3/src/arm_control/src/Hardware:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/arm_3")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/arm_control" TYPE DIRECTORY FILES
    "/home/lin/aloha-ros-dev/arm_follow/arx5_follow3/src/arm_control/launch"
    "/home/lin/aloha-ros-dev/arm_follow/arx5_follow3/src/arm_control/models"
    )
endif()

