from launch import LaunchDescription
from launch.actions import ExecuteProcess

def generate_launch_description():
    # ExecuteProcess for each node
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

    # Create LaunchDescription and add actions
    ld = LaunchDescription()
    ld.add_action(detect_people)
    ld.add_action(detect_rings)
    ld.add_action(tts_node)
    ld.add_action(robot_commander)
    
    return ld
