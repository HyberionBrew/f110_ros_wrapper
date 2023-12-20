from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ros_ftg_agent',
            executable='ros_ftg_agent_node',
            name='ros_ftg_agent_node',
            output='screen',
            # Add specific remappings or parameters for simulation here if needed
        ),
    ])