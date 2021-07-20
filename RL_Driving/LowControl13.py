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

        self.image_sub = rospy.Subscriber("/robot0/camera/image", Image, self.callback)
        # modify code
        self.pub_construct_lane = rospy.Publisher('/robot0/detect/construct_lane', Bool, queue_size=1)
        self.sub_lane_flag = rospy.Subscriber('/robot0/detect/lane_flag', UInt8, self.setLaneFlag, queue_size=1)
        self.sub_scan = rospy.Subscriber("/robot0/scan", LaserScan, self.scanCallBack, queue_size=1)

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

        self.isCarOnLeft = False
        self.isCarOnRight = False
        self.isDangerous = True

        self.recordCount = 1
        self.speedRecord = []
        self.straightCounter=0

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
        self.center = center
        # print('center = desired_center.data: ', desired_center.data)
        print('center: ', center)
        error = center - 480
        Kp = 0.0023
        Kd = 0.007
        # noise = np.random.normal(0,0.03,1)
        angular_z = Kp * error + Kd * (error - self.lastError)
        self.lastError = error
        twist = Twist()
        if (self.isOverTaking or self.isLaneChange or self.isObstacle):
            return
        elif self.isStraightLane == 1:
            twist.linear.x = self.target_vel
            # twist.linear.y = 0
            # twist.linear.z = 0
            # twist.angular.x = 0
            # twist.angular.y = 0
            twist.angular.z = -max(angular_z, -2.0) if angular_z < 0 else -min(angular_z, 2.0)
            self.pub_cmd_vel.publish(twist)
        else:
            self.target_vel = 0.06
            if center > 820 or center < 300:
                print('detect outline')
                twist.linear.x = 0.03
                # twist.angular.z=0
                self.pub_construct_lane.publish(Bool(data=False))
            else:
                self.targetVal = 0.06  # reset the init speed
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

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        self.image_received = True
        self.image = cv_image

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

    def scanCallBack(self, scan_msg):
        print('enter scan call back!')
        if self.scan_counter % 2 != 0:
            self.scan_counter += 1
            return
        else:
            self.scan_counter = 1

        scan = scan_msg
        scan_range = []
        min_range = 0.16

        if min(scan_range) > min_range:
            self.isDangerous = False
        else:
            self.isDangerous = True
        print('The min distance is :', min(scan_range))
        print('Dangerous?', self.isDangerous)

        front_data = [scan_range[i] for i in range(-37, 37)]
        left_data = [scan_range[i] for i in range(-90, -37)]
        right_data = [scan_range[i] for i in range(37, 90)]
        back_data = [scan_range[i] for i in range(135, 255)]
        carOnLeft = [scan_range[i] for i in range(-140, -40)]
        carOnRight = [scan_range[i] for i in range(40, 140)]

        self.front_distance = min(front_data)
        self.left_distance = min(left_data)
        self.right_distance = min(right_data)
        self.back_distance = min(back_data)
        print('the front distance:', self.front_distance)

        if min(carOnLeft) < 0.40:
            self.isCarOnLeft = True
        else:
            self.isCarOnLeft = False
        print('the obstacle on Left:', self.isCarOnLeft)

        if min(carOnRight) < 0.40:
            self.isCarOnRight = True
        else:
            self.isCarOnRight = False
        print('the obstacle on Right:', self.isCarOnRight)

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

    def smoving(self):
        if self.isDangerous:
            print('very dangerous now, need to stop')
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
            if (self.front_distance < 0.40 and self.isCarOnLeft is not True):
                print('enter scan_lane change')
                self.laneChange()
                self.lane_flag = 0
            elif (self.front_distance < 0.40 and self.isCarOnLeft):
                print('not safe, the car need to stop')
                self.MAX_VEL = 0
                stop_cmd = Twist()
                stop_cmd.linear.x = 0
                self.pub_cmd_vel.publish(stop_cmd)
            elif (self.lane_flag == 0) and (self.isLaneChange == False):
                if(self.front_distance > 0 and self.front_distance < 0.35):
                    print('publish stop')
                    self.isDangerous = True
                    self.MAX_VEL = 0
                    stop_cmd = Twist()
                    stop_cmd.linear.x = 0.0
                    stop_cmd.linear.y = 0
                    stop_cmd.linear.z = 0
                    stop_cmd.angular.x = 0
                    stop_cmd.angular.y = 0
                    stop_cmd.angular.z = 0.0
                    self.pub_cmd_vel.publish(stop_cmd)
                elif (self.front_distance > 0.4 and self.front_distance < 0.60):
                    print('publish slow down')
                    self.MAX_SLOW -= 0.005
                    if 450 < self.center < 506:
                        self.MAX_VEL = np.clip(self.MAX_SLOW, 0.07, 0.1)
                    else:
                        self.MAX_VEL = np.clip(self.MAX_SLOW, 0.02, 0.06)
                else:
                    print('enter normal mode')
                    if 450 < self.center < 505:
                        self.MAX_SLOW = 0.06
                    else:
                        self.MAX_SLOW = 0.1
                    self.isObstacle = False
                    self.isDangerous = False
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
                if self.isDangerous:
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

    def main(self):
        while(1):
            smoving()
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('control_lane_robot0')
    node = ControlLane()
    node.main()








