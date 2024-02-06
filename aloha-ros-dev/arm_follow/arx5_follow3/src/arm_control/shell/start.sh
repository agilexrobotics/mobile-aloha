#!/bin/bash
WORKSPACE_PATH=$(readlink --canonicalize ../../../)
SCRIPT_PATH=$(readlink --canonicalize ../script/)

cd ${WORKSPACE_PATH}
catkin_make

source /opt/ros/noetic/setup.bash
source "${WORKSPACE_PATH}/devel/setup.bash"