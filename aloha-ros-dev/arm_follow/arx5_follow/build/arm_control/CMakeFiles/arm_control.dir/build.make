# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lin/aloha-ros-dev/arm_follow/arx5_follow/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lin/aloha-ros-dev/arm_follow/arx5_follow/build

# Include any dependencies generated for this target.
include arm_control/CMakeFiles/arm_control.dir/depend.make

# Include the progress variables for this target.
include arm_control/CMakeFiles/arm_control.dir/progress.make

# Include the compile flags for this target's objects.
include arm_control/CMakeFiles/arm_control.dir/flags.make

arm_control/CMakeFiles/arm_control.dir/src/App/arm_control.cpp.o: arm_control/CMakeFiles/arm_control.dir/flags.make
arm_control/CMakeFiles/arm_control.dir/src/App/arm_control.cpp.o: /home/lin/aloha-ros-dev/arm_follow/arx5_follow/src/arm_control/src/App/arm_control.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lin/aloha-ros-dev/arm_follow/arx5_follow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object arm_control/CMakeFiles/arm_control.dir/src/App/arm_control.cpp.o"
	cd /home/lin/aloha-ros-dev/arm_follow/arx5_follow/build/arm_control && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/arm_control.dir/src/App/arm_control.cpp.o -c /home/lin/aloha-ros-dev/arm_follow/arx5_follow/src/arm_control/src/App/arm_control.cpp

arm_control/CMakeFiles/arm_control.dir/src/App/arm_control.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/arm_control.dir/src/App/arm_control.cpp.i"
	cd /home/lin/aloha-ros-dev/arm_follow/arx5_follow/build/arm_control && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lin/aloha-ros-dev/arm_follow/arx5_follow/src/arm_control/src/App/arm_control.cpp > CMakeFiles/arm_control.dir/src/App/arm_control.cpp.i

arm_control/CMakeFiles/arm_control.dir/src/App/arm_control.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/arm_control.dir/src/App/arm_control.cpp.s"
	cd /home/lin/aloha-ros-dev/arm_follow/arx5_follow/build/arm_control && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lin/aloha-ros-dev/arm_follow/arx5_follow/src/arm_control/src/App/arm_control.cpp -o CMakeFiles/arm_control.dir/src/App/arm_control.cpp.s

arm_control/CMakeFiles/arm_control.dir/src/Hardware/math_ops.cpp.o: arm_control/CMakeFiles/arm_control.dir/flags.make
arm_control/CMakeFiles/arm_control.dir/src/Hardware/math_ops.cpp.o: /home/lin/aloha-ros-dev/arm_follow/arx5_follow/src/arm_control/src/Hardware/math_ops.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lin/aloha-ros-dev/arm_follow/arx5_follow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object arm_control/CMakeFiles/arm_control.dir/src/Hardware/math_ops.cpp.o"
	cd /home/lin/aloha-ros-dev/arm_follow/arx5_follow/build/arm_control && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/arm_control.dir/src/Hardware/math_ops.cpp.o -c /home/lin/aloha-ros-dev/arm_follow/arx5_follow/src/arm_control/src/Hardware/math_ops.cpp

arm_control/CMakeFiles/arm_control.dir/src/Hardware/math_ops.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/arm_control.dir/src/Hardware/math_ops.cpp.i"
	cd /home/lin/aloha-ros-dev/arm_follow/arx5_follow/build/arm_control && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lin/aloha-ros-dev/arm_follow/arx5_follow/src/arm_control/src/Hardware/math_ops.cpp > CMakeFiles/arm_control.dir/src/Hardware/math_ops.cpp.i

arm_control/CMakeFiles/arm_control.dir/src/Hardware/math_ops.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/arm_control.dir/src/Hardware/math_ops.cpp.s"
	cd /home/lin/aloha-ros-dev/arm_follow/arx5_follow/build/arm_control && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lin/aloha-ros-dev/arm_follow/arx5_follow/src/arm_control/src/Hardware/math_ops.cpp -o CMakeFiles/arm_control.dir/src/Hardware/math_ops.cpp.s

arm_control/CMakeFiles/arm_control.dir/src/Hardware/motor.cpp.o: arm_control/CMakeFiles/arm_control.dir/flags.make
arm_control/CMakeFiles/arm_control.dir/src/Hardware/motor.cpp.o: /home/lin/aloha-ros-dev/arm_follow/arx5_follow/src/arm_control/src/Hardware/motor.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lin/aloha-ros-dev/arm_follow/arx5_follow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object arm_control/CMakeFiles/arm_control.dir/src/Hardware/motor.cpp.o"
	cd /home/lin/aloha-ros-dev/arm_follow/arx5_follow/build/arm_control && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/arm_control.dir/src/Hardware/motor.cpp.o -c /home/lin/aloha-ros-dev/arm_follow/arx5_follow/src/arm_control/src/Hardware/motor.cpp

arm_control/CMakeFiles/arm_control.dir/src/Hardware/motor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/arm_control.dir/src/Hardware/motor.cpp.i"
	cd /home/lin/aloha-ros-dev/arm_follow/arx5_follow/build/arm_control && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lin/aloha-ros-dev/arm_follow/arx5_follow/src/arm_control/src/Hardware/motor.cpp > CMakeFiles/arm_control.dir/src/Hardware/motor.cpp.i

arm_control/CMakeFiles/arm_control.dir/src/Hardware/motor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/arm_control.dir/src/Hardware/motor.cpp.s"
	cd /home/lin/aloha-ros-dev/arm_follow/arx5_follow/build/arm_control && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lin/aloha-ros-dev/arm_follow/arx5_follow/src/arm_control/src/Hardware/motor.cpp -o CMakeFiles/arm_control.dir/src/Hardware/motor.cpp.s

arm_control/CMakeFiles/arm_control.dir/src/Hardware/teleop.cpp.o: arm_control/CMakeFiles/arm_control.dir/flags.make
arm_control/CMakeFiles/arm_control.dir/src/Hardware/teleop.cpp.o: /home/lin/aloha-ros-dev/arm_follow/arx5_follow/src/arm_control/src/Hardware/teleop.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lin/aloha-ros-dev/arm_follow/arx5_follow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object arm_control/CMakeFiles/arm_control.dir/src/Hardware/teleop.cpp.o"
	cd /home/lin/aloha-ros-dev/arm_follow/arx5_follow/build/arm_control && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/arm_control.dir/src/Hardware/teleop.cpp.o -c /home/lin/aloha-ros-dev/arm_follow/arx5_follow/src/arm_control/src/Hardware/teleop.cpp

arm_control/CMakeFiles/arm_control.dir/src/Hardware/teleop.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/arm_control.dir/src/Hardware/teleop.cpp.i"
	cd /home/lin/aloha-ros-dev/arm_follow/arx5_follow/build/arm_control && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lin/aloha-ros-dev/arm_follow/arx5_follow/src/arm_control/src/Hardware/teleop.cpp > CMakeFiles/arm_control.dir/src/Hardware/teleop.cpp.i

arm_control/CMakeFiles/arm_control.dir/src/Hardware/teleop.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/arm_control.dir/src/Hardware/teleop.cpp.s"
	cd /home/lin/aloha-ros-dev/arm_follow/arx5_follow/build/arm_control && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lin/aloha-ros-dev/arm_follow/arx5_follow/src/arm_control/src/Hardware/teleop.cpp -o CMakeFiles/arm_control.dir/src/Hardware/teleop.cpp.s

arm_control/CMakeFiles/arm_control.dir/src/utility.cpp.o: arm_control/CMakeFiles/arm_control.dir/flags.make
arm_control/CMakeFiles/arm_control.dir/src/utility.cpp.o: /home/lin/aloha-ros-dev/arm_follow/arx5_follow/src/arm_control/src/utility.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lin/aloha-ros-dev/arm_follow/arx5_follow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object arm_control/CMakeFiles/arm_control.dir/src/utility.cpp.o"
	cd /home/lin/aloha-ros-dev/arm_follow/arx5_follow/build/arm_control && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/arm_control.dir/src/utility.cpp.o -c /home/lin/aloha-ros-dev/arm_follow/arx5_follow/src/arm_control/src/utility.cpp

arm_control/CMakeFiles/arm_control.dir/src/utility.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/arm_control.dir/src/utility.cpp.i"
	cd /home/lin/aloha-ros-dev/arm_follow/arx5_follow/build/arm_control && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lin/aloha-ros-dev/arm_follow/arx5_follow/src/arm_control/src/utility.cpp > CMakeFiles/arm_control.dir/src/utility.cpp.i

arm_control/CMakeFiles/arm_control.dir/src/utility.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/arm_control.dir/src/utility.cpp.s"
	cd /home/lin/aloha-ros-dev/arm_follow/arx5_follow/build/arm_control && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lin/aloha-ros-dev/arm_follow/arx5_follow/src/arm_control/src/utility.cpp -o CMakeFiles/arm_control.dir/src/utility.cpp.s

# Object files for target arm_control
arm_control_OBJECTS = \
"CMakeFiles/arm_control.dir/src/App/arm_control.cpp.o" \
"CMakeFiles/arm_control.dir/src/Hardware/math_ops.cpp.o" \
"CMakeFiles/arm_control.dir/src/Hardware/motor.cpp.o" \
"CMakeFiles/arm_control.dir/src/Hardware/teleop.cpp.o" \
"CMakeFiles/arm_control.dir/src/utility.cpp.o"

# External object files for target arm_control
arm_control_EXTERNAL_OBJECTS =

/home/lin/aloha-ros-dev/arm_follow/arx5_follow/devel/lib/libarm_control.so: arm_control/CMakeFiles/arm_control.dir/src/App/arm_control.cpp.o
/home/lin/aloha-ros-dev/arm_follow/arx5_follow/devel/lib/libarm_control.so: arm_control/CMakeFiles/arm_control.dir/src/Hardware/math_ops.cpp.o
/home/lin/aloha-ros-dev/arm_follow/arx5_follow/devel/lib/libarm_control.so: arm_control/CMakeFiles/arm_control.dir/src/Hardware/motor.cpp.o
/home/lin/aloha-ros-dev/arm_follow/arx5_follow/devel/lib/libarm_control.so: arm_control/CMakeFiles/arm_control.dir/src/Hardware/teleop.cpp.o
/home/lin/aloha-ros-dev/arm_follow/arx5_follow/devel/lib/libarm_control.so: arm_control/CMakeFiles/arm_control.dir/src/utility.cpp.o
/home/lin/aloha-ros-dev/arm_follow/arx5_follow/devel/lib/libarm_control.so: arm_control/CMakeFiles/arm_control.dir/build.make
/home/lin/aloha-ros-dev/arm_follow/arx5_follow/devel/lib/libarm_control.so: arm_control/CMakeFiles/arm_control.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lin/aloha-ros-dev/arm_follow/arx5_follow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX shared library /home/lin/aloha-ros-dev/arm_follow/arx5_follow/devel/lib/libarm_control.so"
	cd /home/lin/aloha-ros-dev/arm_follow/arx5_follow/build/arm_control && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/arm_control.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
arm_control/CMakeFiles/arm_control.dir/build: /home/lin/aloha-ros-dev/arm_follow/arx5_follow/devel/lib/libarm_control.so

.PHONY : arm_control/CMakeFiles/arm_control.dir/build

arm_control/CMakeFiles/arm_control.dir/clean:
	cd /home/lin/aloha-ros-dev/arm_follow/arx5_follow/build/arm_control && $(CMAKE_COMMAND) -P CMakeFiles/arm_control.dir/cmake_clean.cmake
.PHONY : arm_control/CMakeFiles/arm_control.dir/clean

arm_control/CMakeFiles/arm_control.dir/depend:
	cd /home/lin/aloha-ros-dev/arm_follow/arx5_follow/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lin/aloha-ros-dev/arm_follow/arx5_follow/src /home/lin/aloha-ros-dev/arm_follow/arx5_follow/src/arm_control /home/lin/aloha-ros-dev/arm_follow/arx5_follow/build /home/lin/aloha-ros-dev/arm_follow/arx5_follow/build/arm_control /home/lin/aloha-ros-dev/arm_follow/arx5_follow/build/arm_control/CMakeFiles/arm_control.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : arm_control/CMakeFiles/arm_control.dir/depend

