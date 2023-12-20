from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ros_ftg_agent',
            executable='ros_ftg_agent_node',
            name='ros_ftg_agent_node',
            output='screen',
            remappings=[
                ('/odom', 'vesc/odom'),
                # Add more remappings as needed
            ],
            # You can also add parameters here if needed
        ),
    ])