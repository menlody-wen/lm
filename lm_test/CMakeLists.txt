cmake_minimum_required(VERSION 2.8.3)
project(lm_test)
catkin_package()
find_package(catkin REQUIRED)
find_file(Eigen REQUIRED COMPONENTS)
find_package(catkin REQUIRED COMPONENTS
        std_msgs
        sensor_msgs
        roscpp
        pcl_conversions
        pcl_ros
        cv_bridge
        rospy
        )

include_directories( include ${catkin_INCLUDE_DIRS}  ${PCL_INCLUDE_DIRS} )

link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})

add_executable(lm_testNode src/lm.cpp)
target_link_libraries(lm_testNode ${catkin_LIBRARIES} ${PCL_LIBS})
