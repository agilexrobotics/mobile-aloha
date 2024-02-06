cmake_minimum_required(VERSION 3.0.2)
project(arm_control)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CURSES_NEED_NCURSES TRUE)
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  dynamic_reconfigure
  geometry_msgs
  roscpp
  rospy
  std_msgs
  tf
  message_generation
  message_runtime
  kdl_parser
)


# find_package(Curses REQUIRED)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a exec_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  arx5.msg
  JointControl.msg 
  JointInformation.msg 
  ChassisCtrl.msg
  MagicCmd.msg
)

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs#   std_msgs
  # arx
)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
generate_dynamic_reconfigure_options(
  cfg/reconfig.cfg
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
 INCLUDE_DIRS include
#  LIBRARIES arm_control
 CATKIN_DEPENDS dynamic_reconfigure geometry_msgs roscpp rospy std_msgs tf #message_generation message_runtime
#  DEPENDS system_lib
  CATKIN_DEPENDS kdl_parser
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
include
  ${catkin_INCLUDE_DIRS}
  ${roscpp_INCLUDE_DIRS}
)

## Declare a C++ library

add_library(arm_control
   SHARED
   src/App/arm_control.cpp

   src/Hardware/math_ops.cpp
   src/Hardware/motor.cpp
   src/Hardware/teleop.cpp
   src/utility.cpp

  #  src/App/keyboard.cpp
  #  src/App/solve.cpp
  #  src/App/play.cpp
  #  src/libcan/SocketCAN.cpp
  # src/Hardware/can.cpp
 )




## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
# AUX_SOURCE_DIRECTORY(src/libcan ROBOT_LIBCAN_SOURCES)
# AUX_SOURCE_DIRECTORY(src/App ROBOT_APP_SOURCES)
# AUX_SOURCE_DIRECTORY(src/Hardware ROBOT_HARDWARE_SOURCES)
# AUX_SOURCE_DIRECTORY(src ROBOT_SOURCES)

# add_executable(${PROJECT_NAME}_node src/arm_control_node.cpp)
# add_executable(robot ${ROBOT_SOURCES} ${ROBOT_LIBCAN_SOURCES} ${ROBOT_HARDWARE_SOURCES} ${ROBOT_APP_SOURCES})
add_executable(arm1 src/arx_5v.cpp)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
add_dependencies(arm1 ${PROJECT_NAME}_gencfg)

## Specify libraries to link a library or executable target against
# target_link_libraries(robot ${catkin_LIBRARIES} arm_control 
# ${CMAKE_CURRENT_SOURCE_DIR}/src/App/libplay.a
# ${CMAKE_CURRENT_SOURCE_DIR}/src/App/libsolve.a  
# ${CMAKE_CURRENT_SOURCE_DIR}/src/libcan/libcan.a

# kdl_parser

# )

# ## Specify libraries to link a library or executable target against


# target_link_libraries(robot ${catkin_LIBRARIES} arm_control 
# ${CMAKE_CURRENT_SOURCE_DIR}/src/App/lib_arm_play.a
# ${CMAKE_CURRENT_SOURCE_DIR}/src/App/lib_arm_solve.a  
# ${CMAKE_CURRENT_SOURCE_DIR}/src/libcan/lib_arm_can.a

# kdl_parser

# )


if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm.*|arm|aarch64")
    # Link libraries for ARM architecture
    target_link_libraries(arm1 
        ${catkin_LIBRARIES} 
        arm_control 
        # ${CMAKE_CURRENT_SOURCE_DIR}/src/App/lib_arm_play.a 
        # ${CMAKE_CURRENT_SOURCE_DIR}/src/App/lib_arm_solve.a 
        # ${CMAKE_CURRENT_SOURCE_DIR}/src/libcan/libarmcan.so 
        kdl_parser
    )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64)|(^i686)")
    # Link libraries for x86 architecture
    target_link_libraries(arm1 
        ${catkin_LIBRARIES} 
        arm_control 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/App/libplay.so 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/App/libsolve.so 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libcan/libcan.a 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/App/libkey.so
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Hardware/libcan.so 

        kdl_parser
    )
endif()

# message(STATUS "CMAKE_SYSTEM_PROCESSOR =${CMAKE_SYSTEM_PROCESSOR}")
# message(STATUS "CMAKE_SYSTEM_PROCESSOR =${CMAKE_SYSTEM_PROCESSOR}")
# message(STATUS "CMAKE_SYSTEM_PROCESSOR =${CMAKE_SYSTEM_PROCESSOR}")
# message(STATUS "CMAKE_SYSTEM_PROCESSOR =${CMAKE_SYSTEM_PROCESSOR}")
# message(STATUS "CMAKE_SYSTEM_PROCESSOR =${CMAKE_SYSTEM_PROCESSOR}")
# message(STATUS "CMAKE_SYSTEM_PROCESSOR =${CMAKE_SYSTEM_PROCESSOR}")
# message(STATUS "CMAKE_SYSTEM_PROCESSOR =${CMAKE_SYSTEM_PROCESSOR}")
# message(STATUS "CMAKE_SYSTEM_PROCESSOR =${CMAKE_SYSTEM_PROCESSOR}")
# message(STATUS "CMAKE_SYSTEM_PROCESSOR =${CMAKE_SYSTEM_PROCESSOR}")
# message(STATUS "CMAKE_SYSTEM_PROCESSOR =${CMAKE_SYSTEM_PROCESSOR}")
# message(STATUS "CMAKE_SYSTEM_PROCESSOR =${CMAKE_SYSTEM_PROCESSOR}")
# message(STATUS "CMAKE_SYSTEM_PROCESSOR =${CMAKE_SYSTEM_PROCESSOR}")
# message(STATUS "CMAKE_SYSTEM_PROCESSOR =${CMAKE_SYSTEM_PROCESSOR}")
# message(STATUS "CMAKE_SYSTEM_PROCESSOR =${CMAKE_SYSTEM_PROCESSOR}")
# message(STATUS "CMAKE_SYSTEM_PROCESSOR =${CMAKE_SYSTEM_PROCESSOR}")



# target_link_libraries(robot ${catkin_LIBRARIES} arm_control kdl_parser)

# target_link_libraries(arm_control kdl_parser)

#  src/arm_control/CMakeLists.txt
#  src/arm_control/src/App/libplay.so


## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
install(TARGETS arm1
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
install(TARGETS arm1
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(DIRECTORY
  ${CMAKE_CURRENT_SOURCE_DIR}/launch
  ${CMAKE_CURRENT_SOURCE_DIR}/models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)


