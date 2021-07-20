from sensor_msgs.msg import LaserScan

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import numpy as np

most_min = 3.5

class ScanNode():

    def __init__(self):
        #self.pub_cmd_vel = rospy.Publisher('/robot0/cmd_vel', Twist, queue_size = 1)

        self.sub_scan = rospy.Subscriber('/robot0/scan', LaserScan, self.scanCallback, queue_size=1)
        self.counter=0
        self.exeList=[]
        self.counterRecord=0

    def main(self):
        rate = rospy.Rate(1)  # 10hz
        # while not rospy.is_shutdown():
        #     twist = Twist()
        #     twist.linear.x = 0.0
        #     twist.linear.y = 0
        #
        #     twist.linear.z = 0
        #     twist.angular.x = 0
        #     twist.angular.y = 0
        #     twist.angular.z = 0
        #     # pub.publish(hello_str)
        #     self.pub_cmd_vel.publish(twist)
        #     rate.sleep()
        rospy.spin()

    def scanCallback(self,data):
        if self.counter % 1 != 0:
            self.counter += 1
            return
        else:
            self.counter = 1
        self.counterRecord+=1
        #print('enter scanCallback')
        #most_min = 3.5
        global most_min
        scan = data
        scan_range = []
        #print(scan.ranges)
        #print(len(scan.ranges))
        for i in range(len(scan.ranges)):
            #print("scan.ranges: ", scan.ranges[i])
            #print(scan.ranges[i]==0.0)
            if scan.ranges[i]==0.0:
                scan_range.append(3.5)
            else:
                scan_range.append(scan.ranges[i])
        #print(scan_range)

        if np.min(scan_range) < most_min:
            most_min = np.min(scan_range)
        elif np.min(scan_range) < 0.25:
            self.exeList.append(np.min(scan_range))
        print('min_dist: ', np.min(scan_range))
        print('##############  most_min: ', most_min)
        print('recordCounter: ', self.counterRecord, " exelist: ", self.exeList)


if __name__ == '__main__':
    rospy.init_node('scan_55')
    node = ScanNode()
    node.main()
