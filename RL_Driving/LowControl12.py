import rospy
import numpy as np
from std_msgs.msg import Float64, UInt8
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from sensor_msgs.msg import Image, LaserScan
from math import radians
import time
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
import time


class ControlLane():
    def __init__(self):
        self.sub_lane = rospy.Subscriber('/robot0/detect/lane', Float64, self.cbFollowLane, queue_size=1)
        self.sub_max_vel = rospy.Subscriber('/robot0/max_vel', Float64, self.cbGetMaxVel, queue_size=1)
        self.pub_cmd_vel = rospy.Publisher('/robot0/cmd_vel', Twist, queue_size=1)
        # self.sub_overtake = rospy.Subscriber('control/overtake',Bool,self.setOvertake,queue_size=1)
        # self.sub_overtake = rospy.Subscriber('control/laneChange',Bool,self.laneChangeCallBack,queue_size=1)

        self.image_sub = rospy.Subscriber("/robot0/camera/image", Image, self.callback)
        # modify code
        self.pub_construct_lane = rospy.Publisher('/robot0/detect/construct_lane', Bool, queue_size=1)
        self.sub_lane_flag = rospy.Subscriber('/robot0/detect/lane_flag', UInt8, self.setLaneFlag, queue_size=1)
        self.sub_scan = rospy.Subscriber("/robot0/scan", LaserScan, self.scanCallBack, queue_size=1)

        # 2021-2-13
        self.sub_traffic_light = rospy.Subscriber('/robot0/detect/traffic_light', Float64, self.setTrafficLight, queue_size=1)
        self.sub_lane_behavior = rospy.Subscriber('/robot0/lane_behavior', UInt8, self.laneBehaviorCallBack, queue_size=1)

        self.lastError = 0
        self.MAX_VEL = 0.14
        self.overTakeFlag = False
        self.isOverTaking = False
        self.isLaneChange = False
        self.bridge = CvBridge()
        self.isObstacle = False
        self.scan_counter = 1
        self.lane_flag = 1
        self.traffic_light = 0
        self.isStraightLane=0
        self.targetVal=0.06
        self.vel_step=0.01
        self.min_vel=0.02
        self.max_vel=0.14
        self.target_vel=0.06
        self.MAX_SLOW=0.08
        self.center=500
        self.front_distance = 0.25
        self.left_distance = 0.22
        self.right_distance = 0.22
        self.back_distance = 0.25
        self.all_scan_distance = 0.20


        # modification 2020-7-3
        self.recordCount = 1
        self.speedRecord = []
        self.straightCounter=0

        # modification 2020-7-12
        # self.obs_dist = 0
        # self.SAFE_DIST1 = 0.01
        # self.SAFE_DIST2 = 0.8
        # self.save_count=0
        self.pub_construct_lane.publish(Bool(data=False))
        time.sleep(0.2)
        self.pub_construct_lane.publish(Bool(data=False))

        rospy.on_shutdown(self.fnShutDown)

    def cbGetMaxVel(self, max_vel_msg):
        self.MAX_VEL = max_vel_msg.data

    def setLaneFlag(self, lane_flag_msg):
        if self.isLaneChange == False:
            self.lane_flag = lane_flag_msg.data

    def cbFollowLane(self, desired_center):
        self.recordCount = self.recordCount + 1
        center = desired_center.data
        self.center=center
        # print('center = desired_center.data: ', desired_center.data)
        print('center: ', center)
        error = center - 480
        Kp = 0.0023
        Kd = 0.007
        # noise = np.random.normal(0,0.03,1)
        angular_z = Kp * error + Kd * (error - self.lastError)
        self.lastError = error
        twist = Twist()
        if (self.isOverTaking or self.isLaneChange or self.isObstacle or self.traffic_light == 1):
            return
        elif self.isStraightLane==1:
            twist.linear.x = self.target_vel
            # twist.linear.y = 0
            # twist.linear.z = 0
            # twist.angular.x = 0
            # twist.angular.y = 0
            twist.angular.z = -max(angular_z, -2.0) if angular_z < 0 else -min(angular_z, 2.0)
            self.pub_cmd_vel.publish(twist)
        else:
            self.target_vel=0.06
            if center>820 or center<300:
                print('detect outline')
                twist.linear.x=0.03
                #twist.angular.z=0
                self.pub_construct_lane.publish(Bool(data=False))
            else:
                self.targetVal = 0.06 # reset the init speed
                twist.linear.x = min(self.MAX_VEL * ((1 - abs(error) / 500) ** 2.2), 0.2)
                twist.linear.x = twist.linear.x * 1.1
                twist.linear.y = 0
                twist.linear.z = 0
                twist.angular.x = 0
                twist.angular.y = 0
            twist.angular.z = -max(angular_z, -0.25) if angular_z < 0 else -min(angular_z, 0.25)

            self.pub_cmd_vel.publish(twist)
            print('x: ', twist.linear.x, 'z: ', twist.angular.z)

    def laneBehaviorCallBack(self, msg):
        # print('enter laneBehaviorCallBack')
        action = msg.data
        twist = Twist()
        if action == 0:
            print('enter keep speed:0')
            self.target_vel = self.target_vel
        elif action == 1:
            print('enter accelerate speed:1')
            self.target_vel = self.target_vel + self.vel_step
        elif action == 2:
            print('enter decrease speed:2')
            self.target_vel = self.target_vel - self.vel_step
        else:
            print('enter action:3 and lane_flag: ', self.lane_flag)
            if not self.isLaneChange and self.lane_flag == 0:
                self.laneChange()
                self.lane_flag = 1
            else:
                self.target_vel = self.target_vel - self.vel_step
        self.target_vel = np.clip(self.target_vel, self.min_vel, self.max_vel)
        twist.linear.x = self.target_vel
        self.pub_cmd_vel.publish(twist)

    def fnShutDown(self):
        rospy.loginfo("Shutting down. cmd_vel will be 0")
        twist = Twist()
        twist.linear.x = 0
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = 0
        self.pub_cmd_vel.publish(twist)
        time.sleep(0.1)
        self.pub_cmd_vel.publish(twist)

    def setOvertake(self, flag):
        self.overTakeFlag = True
        print('set Overtake')
        self.overTake()

    def laneChangeCallBack(self, msg):
        print('enter laneChangeCallBack')
        self.laneChange()

    def laneChange(self):
        self.isLaneChange = True
        print('enter the lane change mode')
        stop_cmd = Twist()
        stop_cmd.linear.x = 0.0
        stop_cmd.linear.y = 0
        stop_cmd.linear.z = 0
        stop_cmd.angular.x = 0
        stop_cmd.angular.y = 0
        stop_cmd.angular.z = 0.0
        turn_cmd = Twist()
        turn_cmd.linear.x = 0
        turn_cmd.linear.y = 0
        turn_cmd.linear.z = 0
        turn_cmd.angular.x = 0
        turn_cmd.angular.y = 0
        move_cmd = Twist()
        move_cmd.linear.x = 0.13
        move_cmd.linear.y = 0
        move_cmd.linear.z = 0
        move_cmd.angular.x = 0
        move_cmd.angular.y = 0
        move_cmd.angular.z = 0.0
        # step1: stop
        self.pub_cmd_vel.publish(stop_cmd)
        for x in range(0, 4):
            turn_cmd.angular.z = radians(35)  # 70 deg/s in radians/s
            self.pub_cmd_vel.publish(turn_cmd)
            #time.sleep(0.2)
            self.smart_time(0.20)
            # self.r.sleep()
        self.pub_cmd_vel.publish(stop_cmd)
        #time.sleep(0.2)
        self.smart_time(0.20)
        self.pub_cmd_vel.publish(stop_cmd)
        print('publish move cmd')
        self.pub_cmd_vel.publish(move_cmd)
        #time.sleep(0.2)
        self.smart_time(0.20)
        self.pub_cmd_vel.publish(move_cmd)
        while (True):
            print("enter detect distance...")
            dist = self.dist_to_lane(self.image)
            if self.isObstacle:
               print('the obstacle is very close')
            # r.sleep()
            print("dist:", dist)
            if (dist < 10 or dist > 150):
                #time.sleep(.8)
                self.smart_time(0.80)
                self.pub_cmd_vel.publish(stop_cmd)
                #time.sleep(0.5)
                self.smart_time(0.50)
                self.pub_cmd_vel.publish(stop_cmd)
                # self.r.sleep()
                print('publish stop')
                break
            time.sleep(0.8)

        # step4: turn -70
        print('break success')
        for x in range(0, 4):
            turn_cmd.angular.z = radians(-35)
            self.pub_cmd_vel.publish(turn_cmd)
            #time.sleep(0.2)
            self.smart_time(0.20)
            # self.r.sleep()
        self.pub_cmd_vel.publish(stop_cmd)
        #time.sleep(0.1)
        self.smart_time(0.10)

        self.pub_cmd_vel.publish(stop_cmd)
        self.pub_construct_lane.publish(Bool(data=False))
        #time.sleep(0.2)
        self.smart_time(0.20)
        self.pub_construct_lane.publish(Bool(data=False))
        #time.sleep(0.2)
        self.smart_time(0.20)
        self.isLaneChange = False

    def moving(self):
        if self.all_scan_distance > 1:
            print('publish unsafe, that need to stop, dist<0.22')
            self.isObstacle = True
            self.MAX_VEL = 0
            stop_cmd = Twist()
            stop_cmd.linear.x = 0.0
            stop_cmd.linear.y = 0.0
            stop_cmd.linear.z = 0
            stop_cmd.angular.x = 0.0
            stop_cmd.angular.y = 0.0
            stop_cmd.angular.z = 0.0
            self.pub_cmd_vel.publish(stop_cmd)

        elif self.lane_flag == 1:
            if len(front_scan) != 0:
                front_dist = np.min(front_scan)
            else:
                front_dist = 0
            if (front_dist < 0.40 and front_dist > 0.0) and (self.isLaneChange == False) and len(safe_scan) == 0:
                print('enter scan_lanechange')
                self.laneChange()
                self.lane_flag = 0
            elif (front_dist < 0.40 and front_dist > 0.0) and (self.isLaneChange == False) and len(safe_scan) != 0:
                print("not safe")
                self.MAX_VEL = 0
                stop_cmd = Twist()
                stop_cmd.linear.x = 0
                self.pub_cmd_vel.publish(stop_cmd)
        elif (self.lane_flag == 0) and (self.isLaneChange == False):
            if (front_dist > 0 and front_dist < 0.35):
                print('publish left_lane stop')
                self.isObstacle = True
                self.MAX_VEL = 0
                stop_cmd = Twist()
                stop_cmd.linear.x = 0.0
                stop_cmd.linear.y = 0
                stop_cmd.linear.z = 0
                stop_cmd.angular.x = 0
                stop_cmd.angular.y = 0
                stop_cmd.angular.z = 0.0
                self.pub_cmd_vel.publish(stop_cmd)
                print('publish left_lane stop')
            elif (front_dist > 0.40 and front_dist < 0.55):
                print('publish slow down')
                #self.MAX_VEL = 0.04  # slow down
                self.MAX_SLOW-=0.005
                if 450 < self.center < 506:
                    self.MAX_VEL=np.clip(self.MAX_SLOW,0.07,0.1)
                else:
                    self.MAX_VEL=np.clip(self.MAX_SLOW,0.02,0.06)
            else:
                print('enter normal mode')
                if 450<self.center<505:
                    self.MAX_SLOW = 0.06
                else:
                    self.MAX_SLOW = 0.1
                self.isObstacle = False
                self.MAX_VEL = 0.145
        if len(stop_scan) == 0:
            print('enter normal mode again')
            self.isObstacle = False
            self.MAX_VEL = 0.145
    
    def smart_time(self, input_time):
        start_time = time.time()
        if self.isObstacle:
            print('isObstacle is True')
        while(1):
            next_time = time.time()
            if (next_time - start_time) > input_time:
                break
            else:
                if self.isObstacle:
                    print('robot need to stop for illusion')
                    self.MAX_VEL = 0
                    stop_cmd =Twist()
                    stop_cmd.linear.x = 0.0
                    stop_cmd.linear.y = 0.0
                    stop_cmd.linear.z = 0.0
                    stop_cmd.angular.x = 0.0
                    stop_cmd.angular.y = 0.0
                    stop_cmd.angular.z = 0.0
                    self.pub_cmd_vel.publish(stop_cmd)
                    print('robot begin to be stop')

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        self.image_received = True
        self.image = cv_image

    def scanCallBack(self, scan_msg):
        print('enter scan call back!')
        if self.scan_counter % 2 != 0:
            self.scan_counter += 1
            return
        else:
            self.scan_counter = 1
        # front_scan = [scan_msg.ranges[i] for i in range(-28, 6) if scan_msg.ranges[i] != 0]
        if self.lane_flag == 1:
            front_scan = [scan_msg.ranges[i] for i in range(-28, 6) if scan_msg.ranges[i] != 0]
        else:
            front_scan = [scan_msg.ranges[i] for i in range(-28, 30) if scan_msg.ranges[i] != 0]
        safe_scan = [scan_msg.ranges[i] for i in range(35, 120) if scan_msg.ranges[i] > 0 and scan_msg.ranges[i] < 0.30]
        # print(front_scan)
        # print('scan_msg.ranges[0]: ', scan_msg.ranges[0], ' scan_msg.ranges[1]: ', scan_msg.ranges[1], ' scan_msg.ranges[-1] ', scan_msg.ranges[-1], ' scan_msg.ranges[-2]', scan_msg.ranges[-2])
        # front_dist = np.min(front_scan)
        stop_scan = [scan_msg.ranges[i] for i in range(-180,180) if scan_msg.ranges[i] > 0 and scan_msg.ranges[i] < 0.22]
        if len(front_scan) != 0:
            front_dist = np.min(front_scan)
        else:
            front_dist = 0

        self.front_distance = front_dist
        self.all_scan_distance = len(stop_scan)
        print('front_dist: ', front_dist)
        print('the length of stop_scan:', len(stop_scan))


    def dist_to_lane(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(img, 3)
        ret, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        # cv.imwrite("mask_1.jpg",madist
        h, w = mask.shape  # 240, 32dist
        anchor = 3  # anchor of y axis
        out_point = 10  # init value
        for point_x in range(h):
            temp_x = h - point_x - 1
            if self.isObstacle:
               print('robot need to stop in lane change for avoid illusion')
               self.MAX_VEL = 0
               stop_cmd = Twist()
               stop_cmd.linear.x = 0.0
               stop_cmd.linear.y = 0.0
               stop_cmd.linear.z = 0.0
               stop_cmd.angular.x = 0.0
               stop_cmd.angular.y = 0.0
               stop_cmd.angular.z = 0.0
               self.pub_cmd_vel.publish(stop_cmd)
               print('the robot begin to stop now')

            if (temp_x < h - 100):  # more than half of h
                break
            if (mask[temp_x][anchor]) >= 200:  # mask value
                out_point = temp_x
                break
        dist = h - out_point
        return dist

    def setTrafficLight(self, msg):
        self.traffic_light = msg.data
        if self.traffic_light == 1:
            print('receive stop')
            time.sleep(0.2)
            stop_cmd = Twist()
            stop_cmd.linear.x = 0.0
            stop_cmd.linear.y = 0
            stop_cmd.linear.z = 0
            stop_cmd.angular.x = 0
            stop_cmd.angular.y = 0
            stop_cmd.angular.z = 0.0
            self.pub_cmd_vel.publish(stop_cmd)
            time.sleep(0.5)
            self.pub_cmd_vel.publish(stop_cmd)
            print('publish stop')

    def main(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('control_lane_robot0')
    node = ControlLane()
    node.main()

