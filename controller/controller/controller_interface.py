# 基本的に、Indention Testerと同じように使えばいいと思う。
# Controllerのほうは、submoduleにしても良かったけど、一旦断念。Path関係がややこしすぎる。
## コピペで行きます。

# もっと単純化できる案、あります
# 空気圧とか深さとか、全部一つのリクエスト型にいれこむ
# そうすると、いちいちbend_sensor_listenerとかする必要ない
# 今度から、そうするか笑
# classificationとcontrollerをわけるべきやなあ
# シミュレータでは、controllはほぼなくてよかったけど、今回はそうはいかへんもんなあ。
# いや、やっぱり分けたほうがいいやろう。いまから分けよう。
import pickle
import argparse
from rclpy.node import Node

from std_msgs.msg import Float32
from std_msgs.msg import Int32MultiArray
from ttac3_interfaces.srv import TTAC3
from yi2016_utils.node_utils import create_thread
from controller_utils.controller import Controller
from controller_utils.dataset_utils import remove_invalid_data
from controller_utils.dataset_utils import dataset_parser
from ros2serial_interfaces.srv import ROS2SerialSrv

class ControllerInterface(Node):
    def __init__(
        self,
        train_dataset):
        super().__init__('controller_interface')

        self.controller = Controller(train_dataset)
        self.cam_lists = dataset_parser(train_dataset)

def main():
    train_dataset_path = '/home/toshi/dev_ws/src/controller/dataset/20201119123639.pickle'
    with open(train_dataset_path, 'rb') as f:
        train_dataset = pickle.load(f)
    
    train_dataset = remove_invalid_data(train_dataset)
    


if __name__=='__main__':
    main()