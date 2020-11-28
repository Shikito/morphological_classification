import sys
import pickle
import argparse
from time import time
from itertools import product
import random

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32
from std_msgs.msg import Int32MultiArray

from ttac3_interfaces.srv import TTAC3
from yi2016_utils.node_utils import create_thread
from controller_utils.controller import Controller
from controller_utils.dataset_utils import remove_invalid_data
from controller_utils.dataset_utils import dataset_parser
from ros2serial_interfaces.srv import ROS2SerialSrv

from morphological_classification_utils.classifier import Classifier

class ClassifierNode(Node):
    def __init__(
        self,
        home_indent_position,
        train_dataset_path):

        super().__init__('classifier_node')
        
        with open(train_dataset_path, 'rb') as f:
            train_dataset = pickle.load(f)
        self.train_dataset = remove_invalid_data(train_dataset)
        self.cam_lists = dataset_parser(train_dataset)

        self.classifier = Classifier(train_dataset)

        # --- Create Service Client for TTAC3 
        self.cli_ttac3 = self.create_client(TTAC3, 'ttac3')
        while not self.cli_ttac3.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('service not available, waiting again...')
        self.req_ttac3 = TTAC3.Request()

        # --- Create Service Client for Target Air Pressure
        self.cli_target_air_pressure = self.create_client(ROS2SerialSrv, '/terminal/write')
        while not self.cli_target_air_pressure.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('service not available, waiting again...')
        self.req_target_air_pressure = ROS2SerialSrv.Request()

        # --- Create Topic Subscriber for Bend Sensor
        self.sub_bend_sensor = self.create_subscription(
            Float32, 'bend_sensor',
            self.bend_sensor_listener_callback, 10 )

        # --- Main Loop (by Thread)
        update_period = 0.5
        self.count = 0
        self.main_thread = create_thread(update_period, self.update)

    def update(self):
        # if is_random_am is True:
        #     next_a = random.choice(self.a_list)
        #     next_m = random.choice(self.m_list)

        next_a, next_m, _ =\
            self.classifier.calc_next_action_morphology()

        s = self.sensor_sim.sensor_value(next_a, next_m) #TODO
        x = [next_a, next_m, s]

        self.classifier.append_x(x)
        self.classifier.update_posterior_prob()
        estimate_c, max_prob = self.classifier.estimate_c()
        self.classifier.calc_G()
        should_continue = self.classifier.should_continue()

        return {
            'should_continue':should_continue,
            'estimate_c':estimate_c,
            'max_prob':max_prob,
            'selected_a': next_a,
            'selected_m': next_m,
        }
        
        home_position = [125, 85, self.home_indent_position]

        indention_position_list = [
            [125, 85, self.home_indent_position + depth]
            for depth in [3, 6, 9]]
        
        target_air_pressure_list = [100, 125, 150, 175, 200]

        for target_air_pressure, indention_position in product(
            target_air_pressure_list,
            indention_position_list):

            # Set target_air_pressure
            self.get_logger().info(f'Send target_air_pressure : {target_air_pressure}')
            self.req_target_air_pressure.data = str(target_air_pressure)
            response = self.request_service_sync(
                self.cli_target_air_pressure,
                self.req_target_air_pressure
            )
            self.get_logger().info(f'Response : {response.success}')
            
            # Move To Home Position
            self.req_ttac3.xyz_goal = home_position
            self.get_logger().info(f'Send xyz_goal : {self.req_ttac3.xyz_goal}')
            response = self.request_service_sync(
                self.cli_ttac3, self.req_ttac3)
            self.get_logger().info(f'Response : {response.is_success}')
            time.sleep(1)

            # Move To Indent Position
            self.req_ttac3.xyz_goal = indention_position
            self.get_logger().info(f'Send xyz_goal : {self.req_ttac3.xyz_goal}')
            response = self.request_service_sync(
                self.cli_ttac3, self.req_ttac3)
            self.get_logger().info(f'Response : {response.is_success}')
            time.sleep(1)

        while True:
            self.get_logger().info('Complete Task : Please Press Ctrl-C to End')
            
            # Move To Home Position
            self.req_ttac3.xyz_goal = home_position
            response = self.request_service_sync(
                self.cli_ttac3, self.req_ttac3)
            time.sleep(1)
        
    def pub_target_air_pressure_callback(self):
        msg = Float32()
        msg.data = self.target_air_pressure
        self.pub_target_air_pressure.publish(msg)

    def bend_sensor_listener_callback(self, msg):
        self.bend_sensor = msg.data

    def request_service_sync(self, client, req):
        future = client.call_async(req)
        self.get_logger().info('waiting for response')
        while not future.done():
            time.sleep(0.01)
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().info(
                'Service call failed %r' % (e,))

        return response


def arg_parser(argv):
    parser = argparse.ArgumentParser(description='Classifier Node')
    parser.add_argument('-p', '--home_indent_position', type=int)
    parser.add_argument('-t', '--train_dataset_path', type=str,
                        default='/home/toshi/dev_ws/src/morphological_classification/dataset/20201119123639.pickle')
    args = parser.parse_args()
    return args

def main(argv=sys.argv):
    args = arg_parser(argv)
    home_indent_position = args.home_indent_position
    train_dataset_path = args.train_dataset_path

    rclpy.init()
    classifier_node = ClassifierNode(home_indent_position, train_dataset_path,)
    classifier_node.main_thread.start()
    rclpy.spin(classifier_node)

if __name__=='__main__':
    main()