This is the capstone project repo for team _What's In a Name? (WIN)_. Team WIN is composed of the following course participants:

<table>
  <tr>
    <th>Name</th>
    <th>Role / Responsibility</th>
  </tr>
  <tr>
    <td>Helio Perroni Filho</td>
    <td>Team Leader</td>
  </tr>
  <tr>
    <td>Ian Zhang</td>
    <td>Drive-By-Wire (DBW)</td>
  </tr>
  <tr>
    <td>Lucas Souza</td>
    <td>Traffic Sign Detector</td>
  </tr>
  <tr>
    <td>Shay Koren</td>
    <td>Waypoint Updater</td>
  </tr>
  <tr>
    <td>Yong Zhang</td>
    <td>Documentation and Integration Tests</td>
  </tr>
</table>

### Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/self-driving-car-sim/releases/tag/v0.1).

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/carnd_capstone.git
```

2. Install python dependencies
```bash
cd carnd_capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator
```bash
unzip lights_no_cars.zip
cd lights_no_cars
chmod +x ros_test.x86_64
./ros_test.x86_64
```


