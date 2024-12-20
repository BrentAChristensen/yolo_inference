cmake_minimum_required(VERSION 3.5)
project(yolo_inference)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED) # For Python nodes
find_package(std_msgs REQUIRED) # Example dependency, adjust as necessary

# Install Python scripts
install(PROGRAMS
  scripts/yolo_inference_node.py
  DESTINATION lib/${PROJECT_NAME}
)



# Optionally install configuration files, data, etc.
# install(FILES resource/icons/impact_rqt_joint_states.png
#    DESTINATION share/${PROJECT_NAME}/resource/icons
# )

# Add tests (optional)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
