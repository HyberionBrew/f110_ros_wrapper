from setuptools import setup

package_name = 'ros_ftg_agent'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include the launch directory
        ('share/' + package_name + '/launch', ['launch/sim_launch.py', 'launch/hw_launch.py']),
    ],
    install_requires=['setuptools', 'f110_agents'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='fabian.kresse@tuwien.ac.at',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ros_ftg_agent_node = ros_ftg_agent.ros_ftg_agent:main',
        ],
    },
)
