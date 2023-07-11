# robot-calibration
## How to use
### Registration
1. Copy and paste the cut path files in the following directory: ./data_registration/unregistered_cut_path
2. Run the following command to register the cut path:
'''
python .\cut_path_registration.py -i new_axial_path_1.csv -o Unfiltered_axial_01.txt -c CAL00001.csv -t TCP00009.csv -q 8.680086,20.944397,-5.848966,-112.971724,-89.514828,113.590517
'''
check and modify the input, output file name, calibration, tcp number, registration joint angles accordingly
3. Check if the registered cut path is saved in the following directory: ./data_filtering/unfiltered_cut_path

### Filtering
1. Copy and paste the cut path files in the following directory: ./data_registration/unregistered_cut_path
2. Run the following command to register the cut path:
'''
python .\cut_path_registration.py -i new_axial_path_1.csv -o Unfiltered_axial_01.txt -c CAL00001.csv -t TCP00009.csv -q 8.680086,20.944397,-5.848966,-112.971724,-89.514828,113.590517
'''
check and modify the input, output file name, calibration, tcp number, registration joint angles
3. Check if the registered cut path is saved in the following directory: ./data_filtering/unfiltered_cut_path

## Reference
The level 2 calibration of robot kinematics using Robotics Toolbox Python

Paper: https://ieeexplore.ieee.org/document/9561366

Code: https://github.com/petercorke/robotics-toolbox-python

The level 2 calibration of robot kinematics using local / minimal POE

Paper: Local POE Model for Robot Kinematic Calibration

Paper: A Minimal POE-based Model for Robotic Kinematic Calibration with Only Position Measurements