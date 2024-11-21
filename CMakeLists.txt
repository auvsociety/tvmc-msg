cmake_minimum_required(VERSION 3.0.2)
project(rose_tvmc_msg)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs 
  message_generation
)


add_message_files(
  DIRECTORY msg
  FILES
  Command.msg
  ControlMode.msg
  CurrentPoint.msg
  DoF.msg
  PidConstants.msg
  PidLimits.msg
  TargetPoint.msg
  Thrust.msg
  LEDControl.msg
)

catkin_python_setup()

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs 
)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs geometry_msgs message_runtime 
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
