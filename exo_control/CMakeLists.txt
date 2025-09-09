cmake_minimum_required(VERSION 3.8)
project(exo_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(exo_interfaces REQUIRED)

install(PROGRAMS
  exo_control/two_boards_exo_control.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  exo_control/admittance_controller.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  exo_control/stoop_trajs.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  exo_control/traj_client.py
  DESTINATION lib/${PROJECT_NAME}
)


ament_package()



