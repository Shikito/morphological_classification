import sys
import pickle
import argparse
import random
from datetime import datetime
from pathlib import Path
from time import time
from itertools import product

import numpy as np
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32
from ttac3_interfaces.srv import TTAC3
from ros2serial_interfaces.srv import ROS2SerialSrv

from yi2016_utils.node_utils import create_thread
from controller_utils.controller import Controller
from controller_utils.dataset_utils import remove_invalid_data
from controller_utils.dataset_utils import dataset_parser
from morphological_classification_utils.classifier import Classifier

class ClassifierNode(Node):
    def __init__(
        self,
        true_c,
        home_indent_position,
        train_dataset_path,
        log_file_path,
        is_random_am=False, # 提案手法を使わず、randomなamにしたい場合はTrueにしてください.
        bend_sensor_log_len=100, # 平均取る前の曲げセンサのセンサ値の個数
        ):

        self.true_c = true_c
        self.home_indent_position = home_indent_position
        self.train_dataset_path = train_dataset_path
        self.log_file_path = log_file_path
        self.is_random_am = is_random_am

        # クラス分類の記録に必要な変数
        self.step = 0
        self.classification_log_list = [] 
        
        super().__init__('classifier_node')
        
        # --- Load Train Dataset 
        with open(self.train_dataset_path, 'rb') as f:
            train_dataset = pickle.load(f)
        self.train_dataset = remove_invalid_data(train_dataset)
        self.cam_lists = dataset_parser(train_dataset)

        # 
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
        def request_target_air_pressure(target_air_pressure):
            self.get_logger().info(f'Send target_air_pressure : {target_air_pressure}')
            self.req_target_air_pressure.data = str(target_air_pressure)
            response = self.request_service_sync(
                self.cli_target_air_pressure,
                self.req_target_air_pressure
            )
            self.get_logger().info(f'Response : {response.success}')

        def request_xyz_goal(xyz_goal):
            self.req_ttac3.xyz_goal = xyz_goal
            self.get_logger().info(f'Send xyz_goal : {self.req_ttac3.xyz_goal}')
            response = self.request_service_sync(
                self.cli_ttac3, self.req_ttac3)
            self.get_logger().info(f'Response : {response.is_success}')
            time.sleep(1)

        # 3軸ステージをホームポジションへ
        home_position = [125, 85, self.home_indent_position]
        request_xyz_goal(home_position)

        # 次のactionとmorphology(next_a, next_m)を決定
        if self.is_random_am is True:
            next_a = random.choice(self.a_list)
            next_m = random.choice(self.m_list)
        else:
            next_a, next_m, _ =\
                self.classifier.calc_next_action_morphology()

        # next_a, next_mへシステムを移行
        indent_position = home_position 
        indent_position[-1] += next_a # next_aはhomepositionを基準とした深さ情報
        target_air_pressure = next_m  # next_mは目標空気圧の値。なので、そのまま。
        request_xyz_goal(indent_position)
        request_target_air_pressure(target_air_pressure)

        # センサ値sを取得
        self.bend_sensor_log = [] # 初期化
        while len(self.bend_sensor_log) < 100:
            pass
        s_mean = np.mean(self.bend_sensor_log)
        
        # データを保存
        x = [next_a, next_m, s_mean]
        self.classifier.append_x(x)

        # クラス推定
        self.classifier.update_posterior_prob()
        estimate_c, max_prob = self.classifier.estimate_c()

        # 終了判定
        self.classifier.calc_G()
        should_continue = self.classifier.should_continue()

        log = {
            'should_continue':should_continue,
            'estimate_c':estimate_c,
            'max_prob':max_prob,
            'selected_a': next_a,
            'selected_m': next_m,
        }

        self.get_logger().info('--------------------------------------')
        self.get_logger().info(f'step            : {self.step}')
        self.get_logger().info(f'estimate_c      : {log["estimate_c"]}')
        self.get_logger().info(f'true_c          : {self.true_c}')
        self.get_logger().info(f'max_prob        : {log["max_prob"]}')
        self.get_logger().info(f'selected_a      : {log["selected_a"]}')
        self.get_logger().info(f'selected_m      : {log["selected_m"]}')
        self.get_logger().info(f'should_continue : {log["should_continue"]}')
        self.get_logger().info('--------------------------------------')
        
        self.classification_log_list.append(log)

        if log['should_continue']:
            self.step += 1
        else:
            with open(str(self.log_file_path), 'wb') as f:
                pickle.dump(self.classification_log_list, f)
                self.get_logger().info('--------------------------------------')
                self.get_logger().info(f'    *********Finish***********       ')
                self.get_logger().info(f'Is Success      : {log["estimate_c"] == self.true_c}')
                self.get_logger().info(f'true_c          : {self.true_c}')
                self.get_logger().info(f'estimate_c      : {log["estimate_c"]}')
                self.get_logger().info(f'step num        : {self.step}')
                self.get_logger().info('--------------------------------------')

            while not log['should_continue']:
                self.get_logger().info('Complete Task : Please Press Ctrl-C to End')
                self.get_logger()
                request_xyz_goal(home_position)
                time.sleep(1)
        
    def pub_target_air_pressure_callback(self):
        msg = Float32()
        msg.data = self.target_air_pressure
        self.pub_target_air_pressure.publish(msg)

    def bend_sensor_listener_callback(self, msg):
        self.bend_sensor = msg.data
        self.bend_sensor_log.append(msg.data)
        self.bend_sensor_log = self.bend_sensor_log[:100] # 100個登録可能。個数は決め打ち。

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
    parser.add_argument('-w', '--true_w', type=str)
    parser.add_argument('-r', '--true_r', type=str)
    parser.add_argument('-p', '--home_indent_position', type=int)
    parser.add_argument('-i', '--is_random_am', type=int)
    parser.add_argument('-t', '--train_dataset_path', type=str,
                        default='/home/toshi/dev_ws/src/morphological_classification/dataset/20201119123639.pickle')
    parser.add_argument('-l', '--classification_log_dir', type=str,
                        default='/home/toshi/dev_ws/pickle/')
    args = parser.parse_args()
    return args

def main(argv=sys.argv):
    args = arg_parser(argv)

    # Define name of log file 
    dt_now = datetime.now().strftime('%Y%m%d%H%M%S')
    log_dir = Path(args.classification_log_dir)
    file_name = f'truec-{args.true_c}__{dt_now}.pickle'
    log_file_path = log_dir / file_name

    true_c = f'width_{args.w}_radious_{args.r}'

    rclpy.init()
    classifier_node = ClassifierNode(
        true_c=args.true_c,
        home_indent_position=args.home_indent_position,
        train_dataset=args.train_dataset_path,
        log_file_path=log_file_path)

    classifier_node.main_thread.start()
    rclpy.spin(classifier_node)


if __name__=='__main__':
    main()