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
CMAKE_SOURCE_DIR = /home/lin/aloha-ros-dev/arm_follow/arx5_follow4/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lin/aloha-ros-dev/arm_follow/arx5_follow4/build

# Utility rule file for arm_control_gencfg.

# Include the progress variables for this target.
include arm_control/CMakeFiles/arm_control_gencfg.dir/progress.make

arm_control/CMakeFiles/arm_control_gencfg: /home/lin/aloha-ros-dev/arm_follow/arx5_follow4/devel/include/arm_control/reconfigConfig.h
arm_control/CMakeFiles/arm_control_gencfg: /home/lin/aloha-ros-dev/arm_follow/arx5_follow4/devel/lib/python3/dist-packages/arm_control/cfg/reconfigConfig.py


/home/lin/aloha-ros-dev/arm_follow/arx5_follow4/devel/include/arm_control/reconfigConfig.h: /home/lin/aloha-ros-dev/arm_follow/arx5_follow4/src/arm_control/cfg/reconfig.cfg
/home/lin/aloha-ros-dev/arm_follow/arx5_follow4/devel/include/arm_control/reconfigConfig.h: /opt/ros/noetic/share/dynamic_reconfigure/templates/ConfigType.py.template
/home/lin/aloha-ros-dev/arm_follow/arx5_follow4/devel/include/arm_control/reconfigConfig.h: /opt/ros/noetic/share/dynamic_reconfigure/templates/ConfigType.h.template
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/lin/aloha-ros-dev/arm_follow/arx5_follow4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating dynamic reconfigure files from cfg/reconfig.cfg: /home/lin/aloha-ros-dev/arm_follow/arx5_follow4/devel/include/arm_control/reconfigConfig.h /home/lin/aloha-ros-dev/arm_follow/arx5_follow4/devel/lib/python3/dist-packages/arm_control/cfg/reconfigConfig.py"
	cd /home/lin/aloha-ros-dev/arm_follow/arx5_follow4/build/arm_control && ../catkin_generated/env_cached.sh /home/lin/aloha-ros-dev/arm_follow/arx5_follow4/build/arm_control/setup_custom_pythonpath.sh /home/lin/aloha-ros-dev/arm_follow/arx5_follow4/src/arm_control/cfg/reconfig.cfg /opt/ros/noetic/share/dynamic_reconfigure/cmake/.. /home/lin/aloha-ros-dev/arm_follow/arx5_follow4/devel/share/arm_control /home/lin/aloha-ros-dev/arm_follow/arx5_follow4/devel/include/arm_control /home/lin/aloha-ros-dev/arm_follow/arx5_follow4/devel/lib/python3/dist-packages/arm_control

/home/lin/aloha-ros-dev/arm_follow/arx5_follow4/devel/share/arm_control/docs/reconfigConfig.dox: /home/lin/aloha-ros-dev/arm_follow/arx5_follow4/devel/include/arm_control/reconfigConfig.h
	@$(CMAKE_COMMAND) -E touch_nocreate /home/lin/aloha-ros-dev/arm_follow/arx5_follow4/devel/share/arm_control/docs/reconfigConfig.dox

/home/lin/aloha-ros-dev/arm_follow/arx5_follow4/devel/share/arm_control/docs/reconfigConfig-usage.dox: /home/lin/aloha-ros-dev/arm_follow/arx5_follow4/devel/include/arm_control/reconfigConfig.h
	@$(CMAKE_COMMAND) -E touch_nocreate /home/lin/aloha-ros-dev/arm_follow/arx5_follow4/devel/share/arm_control/docs/reconfigConfig-usage.dox

/home/lin/aloha-ros-dev/arm_follow/arx5_follow4/devel/lib/python3/dist-packages/arm_control/cfg/reconfigConfig.py: /home/lin/aloha-ros-dev/arm_follow/arx5_follow4/devel/include/arm_control/reconfigConfig.h
	@$(CMAKE_COMMAND) -E touch_nocreate /home/lin/aloha-ros-dev/arm_follow/arx5_follow4/devel/lib/python3/dist-packages/arm_control/cfg/reconfigConfig.py

/home/lin/aloha-ros-dev/arm_follow/arx5_follow4/devel/share/arm_control/docs/reconfigConfig.wikidoc: /home/lin/aloha-ros-dev/arm_follow/arx5_follow4/devel/include/arm_control/reconfigConfig.h
	@$(CMAKE_COMMAND) -E touch_nocreate /home/lin/aloha-ros-dev/arm_follow/arx5_follow4/devel/share/arm_control/docs/reconfigConfig.wikidoc

arm_control_gencfg: arm_control/CMakeFiles/arm_control_gencfg
arm_control_gencfg: /home/lin/aloha-ros-dev/arm_follow/arx5_follow4/devel/include/arm_control/reconfigConfig.h
arm_control_gencfg: /home/lin/aloha-ros-dev/arm_follow/arx5_follow4/devel/share/arm_control/docs/reconfigConfig.dox
arm_control_gencfg: /home/lin/aloha-ros-dev/arm_follow/arx5_follow4/devel/share/arm_control/docs/reconfigConfig-usage.dox
arm_control_gencfg: /home/lin/aloha-ros-dev/arm_follow/arx5_follow4/devel/lib/python3/dist-packages/arm_control/cfg/reconfigConfig.py
arm_control_gencfg: /home/lin/aloha-ros-dev/arm_follow/arx5_follow4/devel/share/arm_control/docs/reconfigConfig.wikidoc
arm_control_gencfg: arm_control/CMakeFiles/arm_control_gencfg.dir/build.make

.PHONY : arm_control_gencfg

# Rule to build all files generated by this target.
arm_control/CMakeFiles/arm_control_gencfg.dir/build: arm_control_gencfg

.PHONY : arm_control/CMakeFiles/arm_control_gencfg.dir/build

arm_control/CMakeFiles/arm_control_gencfg.dir/clean:
	cd /home/lin/aloha-ros-dev/arm_follow/arx5_follow4/build/arm_control && $(CMAKE_COMMAND) -P CMakeFiles/arm_control_gencfg.dir/cmake_clean.cmake
.PHONY : arm_control/CMakeFiles/arm_control_gencfg.dir/clean

arm_control/CMakeFiles/arm_control_gencfg.dir/depend:
	cd /home/lin/aloha-ros-dev/arm_follow/arx5_follow4/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lin/aloha-ros-dev/arm_follow/arx5_follow4/src /home/lin/aloha-ros-dev/arm_follow/arx5_follow4/src/arm_control /home/lin/aloha-ros-dev/arm_follow/arx5_follow4/build /home/lin/aloha-ros-dev/arm_follow/arx5_follow4/build/arm_control /home/lin/aloha-ros-dev/arm_follow/arx5_follow4/build/arm_control/CMakeFiles/arm_control_gencfg.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : arm_control/CMakeFiles/arm_control_gencfg.dir/depend

