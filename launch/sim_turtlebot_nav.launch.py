from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

# Get package directory
pkg_dis_tutorial3 = get_package_share_directory('dis_tutorial3')

# Declare launch arguments
ARGUMENTS = [
    DeclareLaunchArgument('namespace', default_value='',
                          description='Robot namespace'),
    DeclareLaunchArgument('rviz', default_value='true',
                          choices=['true', 'false'], description='Start rviz.'),
    DeclareLaunchArgument('world', default_value='task1',
                          description='Ignition World'),
    DeclareLaunchArgument('model', default_value='standard',
                          choices=['standard', 'lite'],
                          description='Turtlebot4 Model'),
    DeclareLaunchArgument('map', default_value=PathJoinSubstitution(
                          [pkg_dis_tutorial3, 'maps', 'map.yaml']),
                          description='Full path to map yaml file to load'),
]

for pose_element in ['x', 'y', 'z', 'yaw']:
    ARGUMENTS.append(DeclareLaunchArgument(pose_element, default_value='0.0',
                     description=f'{pose_element} component of the robot pose.'))


def generate_launch_description():
    # Directories for launch files
    package_dir = get_package_share_directory('dis_tutorial3')

    ignition_launch = PathJoinSubstitution(
        [package_dir, 'launch', 'ignition.launch.py'])
    robot_spawn_launch = PathJoinSubstitution(
        [package_dir, 'launch', 'turtlebot4_spawn.launch.py'])
    localization_launch = PathJoinSubstitution(
        [pkg_dis_tutorial3, 'launch', 'localization.launch.py'])
    nav2_launch = PathJoinSubstitution(
        [pkg_dis_tutorial3, 'launch', 'nav2.launch.py'])

    # Launch configurations
    namespace = LaunchConfiguration('namespace')
    map_file = LaunchConfiguration('map')
    use_sim_time = LaunchConfiguration('use_sim_time')
    x, y, z = LaunchConfiguration('x'), LaunchConfiguration('y'), LaunchConfiguration('z')
    yaw = LaunchConfiguration('yaw')

    # Start Ignition simulation
    ignition = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([ignition_launch]),
        launch_arguments=[
            ('world', LaunchConfiguration('world'))
        ]
    )

    # Robot spawn configuration
    robot_spawn = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([robot_spawn_launch]),
        launch_arguments=[
            ('namespace', LaunchConfiguration('namespace')),
            ('rviz', LaunchConfiguration('rviz')),
            ('x', LaunchConfiguration('x')),
            ('y', LaunchConfiguration('y')),
            ('z', LaunchConfiguration('z')),
            ('yaw', LaunchConfiguration('yaw'))
        ]
    )

    # Localization
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([localization_launch]),
        launch_arguments=[
            ('namespace', namespace),
            ('use_sim_time', use_sim_time),
            ('map', map_file),
        ]
    )

    # Navigation (Nav2)
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([nav2_launch]),
        launch_arguments=[
            ('namespace', namespace),
            ('use_sim_time', use_sim_time)
        ]
    )

    # New Nodes: TTS, detect_rings, detect_people, robot_commander
    detect_people = ExecuteProcess(
        cmd=['ros2', 'run', 'dis_tutorial3', 'detect_people.py'],
        output='screen'
    )
    
    detect_rings = ExecuteProcess(
        cmd=['ros2', 'run', 'dis_tutorial3', 'detect_rings.py'],
        output='screen'
    )
    
    tts_node = ExecuteProcess(
        cmd=['ros2', 'run', 'dis_tutorial3', 'TTS_node.py'],
        output='screen'
    )
    
    robot_commander = ExecuteProcess(
        cmd=['ros2', 'run', 'dis_tutorial3', 'robot_commander.py'],
        output='screen'
    )

    # Create launch description and add actions
    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(ignition)
    ld.add_action(robot_spawn)
    ld.add_action(localization)
    ld.add_action(nav2)
    ld.add_action(detect_people)
    ld.add_action(detect_rings)
    ld.add_action(tts_node)
    ld.add_action(robot_commander)
    
    return ld
