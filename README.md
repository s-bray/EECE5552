Make a new directory first, I am using the directory as "spot_ws".

Then clone this repo into the folder "spot_ws"

Open terminal :
cd ~/spot_ws
colcon build

to run the gazebo simulation
cd ~/spot_ws
source install/setup.bash
ros2 launch spot_bringup spot.gz.launch.py


to publish a simple velocity to the /cmd_vel topic, use
cd ~/spot_ws
source install/setup.bash
ros2 topic pub -r 10 /cmd_vel geometry_msgs/Twist '{linear: {x: <use any value between 0 and 1>}, angular: {z: <use any value between 0 and 0.5>}}'



