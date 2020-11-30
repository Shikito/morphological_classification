import os
import sys
import argparse
from datetime import datetime as dt

import launch
from ros2launch.api.api import parse_launch_arguments
from launch import LaunchDescription
from launch_ros.actions import Node


def get_odict_item(od_items, key):
    for item in od_items:
        if item[0] == key:
            return item[1]
    return None


def generate_launch_description():
    launch_argments = [text for text in sys.argv if ':=' in text]
    parsed_launch_argments = parse_launch_arguments(launch_argments)
    print(parsed_launch_argments)

    return LaunchDescription([
        launch.actions.ExecuteProcess(
            cmd=[
                'ros2', 'run', 'morphological_classification', 'classifier',
                '-w', get_odict_item(parsed_launch_argments, 'true_w'),
                '-r', get_odict_item(parsed_launch_argments, 'true_r'),
                '-p', get_odict_item(parsed_launch_argments, 'home_indent_position'),
                '-i', get_odict_item(parsed_launch_argments, 'is_random_am'),
            ]
        ),
        launch.actions.ExecuteProcess(
            cmd=[
                'ros2', 'run', 'ros2serial', 'main',
                '-p', '/dev/ttyACM0',
                '-b', '115200',
                '-n', 'bend_sensor'
            ]
        ),
        launch.actions.ExecuteProcess(
            cmd=[
                'ros2', 'run', 'ros2serial', 'main',
                '-p', '/dev/ttyACM1',
                '-b', '115200',
                '-n', 'terminal'
            ]
        ),
        launch.actions.ExecuteProcess(
            cmd=[
                'ros2', 'run', 'mf_driver', 'bend_sensor_driver'
            ]
        ),
        launch.actions.ExecuteProcess(
            cmd=[
                'ros2', 'run', 'mf_driver', 'terminal_driver'
            ]
        ),
    ])