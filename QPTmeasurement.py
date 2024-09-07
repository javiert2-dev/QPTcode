import time
import clr
import os
import numpy as np
from scipy.optimize import minimize
import sys
import csv

# Update references accordingly after installing the necessary software. Look at the group's QPT library.
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.DeviceManagerCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.GenericMotorCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\ThorLabs.MotionControl.KCube.DCServoCLI.dll")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'C:\\Users\\USER\\KwiatLabSoftwatre\\TomographyCode\\Quantum-Tomography\\src\\QuantumTomography')))


from Thorlabs.MotionControl.DeviceManagerCLI import *
from Thorlabs.MotionControl.GenericMotorCLI import *
from Thorlabs.MotionControl.GenericMotorCLI import KCubeMotor
from Thorlabs.MotionControl.GenericMotorCLI.ControlParameters import JogParametersBase
from Thorlabs.MotionControl.KCube.DCServoCLI import *
from System import Decimal
from datetime import datetime
from ctypes import cdll, c_long, c_ulong, c_uint32, byref, create_string_buffer, c_bool, c_char_p, c_int, c_int16, c_double, sizeof, c_voidp
from TLPMX import TLPMX
from TLPMX import TLPM_DEFAULT_CHANNEL
import subprocess
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from QuantumTomographyCode import *




class KCubeActuatorController:
    def __init__(self, serial_no):
        self.serial_no = serial_no
        self.device = None
        self.initialize_device()

    def initialize_device(self):
        SimulationManager.Instance.InitializeSimulations()
        DeviceManagerCLI.BuildDeviceList()
        self.device = KCubeDCServo.CreateKCubeDCServo(self.serial_no)
        self.device.Connect(self.serial_no)
        time.sleep(0.25)
        self.device.StartPolling(250)
        time.sleep(0.25)
        self.device.EnableDevice()
        time.sleep(0.25)

        if not self.device.IsSettingsInitialized():
            self.device.WaitForSettingsInitialized(10000)
            assert self.device.IsSettingsInitialized() is True

        m_config = self.device.LoadMotorConfiguration(
            self.serial_no,
            DeviceConfiguration.DeviceSettingsUseOptionType.UseFileSettings
        )
        m_config.DeviceSettingsName = "PRMTZ8"
        m_config.UpdateCurrentConfiguration()
        self.device.SetSettings(self.device.MotorDeviceSettings, True, False)
        self.device.Home(60000)
        print(f"Device {self.serial_no} initialized and homed")

    def move_to_position(self, position):
        """
        Moves the WP by the desired amount. The input must be in degrees and must be positive.
        ----------
        position: Angle in degrees, must be positive.

        Returns
        -------
        """
        try:
            d = Decimal(position)
            self.device.MoveTo(d, 20000)
            time.sleep(0.1)
            print(f"Device {self.serial_no} now at position {self.device.Position}")
        except Exception as e:
            print(f"Error moving device {self.serial_no} to position {position}: {e}")

    def close_device(self):
        try:
            self.device.Disconnect()
        except Exception as e:
            print(f"Error closing device {self.serial_no}: {e}")
        finally:
            SimulationManager.Instance.UninitializeSimulations()

def read_power(meter):
    """
    Returns the power meter's reading in microwatts.
    """
    power = c_double()
    meter.measPower(byref(power), TLPM_DEFAULT_CHANNEL)
    return power.value * 1000000  # Convert to µW

def objective_function(angles, QWP, HWP, power_meter):
    """
    This function will be minimized by the scipi algorithm. Therefore power is minimized.
    Parameters
    ----------
    angles : the angles the QWP and HWP should move to.
    QWP: QWP to be rotated.
    HWP: HWP to be rotated.
    power_meter: Input the power meter you want to minimze the power on.

    Returns
    -------
    power: power reading of chosen power meter.
    """

    QWP_angle, HWP_angle = angles
    QWP.move_to_position(QWP_angle)
    HWP.move_to_position(HWP_angle)
    time.sleep(0.9)  # Allow time for measurement to converge
    power = read_power(power_meter)
    return power

def process_tomography(QWPgen, HWPgen, QWPproj, HWPproj, power_meter, gen_positions, proj_positions):
    """
    This function rotates the automated WPs to take the 36 measurements that are necessary for the QPT.
    ----------
    QWPgen : QWP used to generate eigenstates
    HWPgen: HWP used to generate eigenstates
    QWPproj: QWP used to project eigenstates
    HWPproj: QWP used to project eigenstates
    power_meter : power meter used to take the measurements
    gen_positions : 6*3 array that describes the angles the generation WPs must go to to produce the eigenstates. Each row has the following format ("H", QWPgenH, HWPgenH)
    proj_positions : 6*3 array that describes the angles the projection WPs must go to to produce the eigenstates. Each row has the following format ("H", QWPgenH, HWPgenH)

    Returns
    -------
    results : A 6*6 array where the each row corresponds to the generation of H, V, D, A, R and L respectively and each column corresponds to the projection of H, V, D, A, R and L respectively
    """
    # results = []
    results = np.zeros((6, 6))

    for i, gen_pos in enumerate(gen_positions):
        # Move to generation position
        QWPproj.move_to_position(gen_pos[1])
        HWPproj.move_to_position(gen_pos[2])
        print(f"Eigenstate generated is {gen_pos[0]}, {gen_pos[1]}, {gen_pos[2]}")

        for j, proj_pos in enumerate(proj_positions):
            print(f"Projection state is {proj_pos[0]}")
            # Move to projection position
            QWPgen.move_to_position(proj_pos[1])
            HWPgen.move_to_position(proj_pos[2])
            
            # Read power values
            power_reading = read_power(power_meter)
            print(f"{gen_pos[0]}{proj_pos[0]} power is {power_reading}")            
            results[i, j] = power_reading 
            
    return results

def remove_file_if_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File {file_path} removed.")
    else:
        print(f"File {file_path} does not exist.")

def write_results_to_csv(results, data_file):
    labels = ['H', 'V', 'D', 'A', 'R', 'L']
    entries = []

    # Generate the entries
    for i, row_label in enumerate(labels):
        for j, col_label in enumerate(labels):
            entry = [f"{row_label}{col_label}o", results[i][j]]
            entries.append(entry)  # Use a list with two elements

    # Write to CSV file
    try:
        remove_file_if_exists(data_file)
        with open(data_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(entries)  # Use writerows to handle lists of lists
        print(f"File '{data_file}' created successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    start_time = time.time()

    # Initialize power meters
    meter1 = TLPMX()
    meter2 = TLPMX()
    deviceCount1 = c_uint32()
    deviceCount2 = c_uint32()
    
    meter1.findRsrc(byref(deviceCount1))
    meter2.findRsrc(byref(deviceCount2))

    # Check to see if both of the power meters are connected
    if deviceCount1.value == 0 or deviceCount2.value == 0:
        print('No connected meters')
        return

    resourceName1 = create_string_buffer(1024)
    resourceName2 = create_string_buffer(1024)
    meter1.getRsrcName(c_int(0), resourceName1)
    meter2.getRsrcName(c_int(1), resourceName2)

    meter1.open(resourceName1, c_bool(True), c_bool(True))
    meter2.open(resourceName2, c_bool(True), c_bool(True))
    meter1.setWavelength(c_double(635.0), TLPM_DEFAULT_CHANNEL)
    meter2.setWavelength(c_double(635.0), TLPM_DEFAULT_CHANNEL)

    # Initialize waveplate controllers
    QWPgenSerial = "27501110"
    QWPgen = KCubeActuatorController(QWPgenSerial)
    HWPgenSerial = "27002915"
    HWPgen = KCubeActuatorController(HWPgenSerial)
    QWPprojSerial = "27501224"
    QWPproj = KCubeActuatorController(QWPprojSerial)
    HWPprojSerial = "27252048"
    HWPproj = KCubeActuatorController(HWPprojSerial)

    # -- SET OF ANGLES USED TO GENERATE THE EIGENSTATES --
    QWPgenH = 48.271
    HWPgenH = 31.724
    QWPgenV = 92.193
    HWPgenV = 118.204
    QWPgenD = 115.695
    HWPgenD = 75.391
    QWPgenA = 70.073
    HWPgenA = 74.439
    QWPgenR = 41.294
    HWPgenR = 67.529
    QWPgenL = 81.388
    HWPgenL = 57.813

    # -- SET OF ANGLES USED TO PROJECT ONTO THE EIGENSTATES: ACHIEVED BY MINIMIZING PROJECTION ONTO THE ORTHOGONAL STATE --

    QWPprojH = 30.781
    HWPprojH = 74.146
    QWPprojV = 32.737
    HWPprojV = 30.893
    QWPprojD = 78.669
    HWPprojD = 54.032
    QWPprojA = 73.687
    HWPprojA = 96.089
    QWPprojR = 32.059
    HWPprojR = 9.908
    QWPprojL = 18.821
    HWPprojL = 45.921

    QWPgen_positions = [
    ("H", QWPgenH, HWPgenH),
    ("V", QWPgenV, HWPgenV),
    ("D", QWPgenD, HWPgenD),
    ("A", QWPgenA, HWPgenA),
    ("R", QWPgenR, HWPgenR),
    ("L", QWPgenL, HWPgenL)
]

    QWPproj_positions = [
    ("H", QWPprojH, HWPprojH),
    ("V", QWPprojV, HWPprojV),
    ("D", QWPprojD, HWPprojD),
    ("A", QWPprojA, HWPprojA),
    ("R", QWPprojR, HWPprojR),
    ("L", QWPprojL, HWPprojL),
]
    try:

        # Employ move_to_position to set WPs to certain angles and use the minimize function to find the correct set of angles for the generation and projection of each eigenstate

        QWPgen.move_to_position(QWPgenH)
        HWPgen.move_to_position(HWPgenH)
        QWPproj.move_to_position(QWPprojH)
        HWPproj.move_to_position(HWPprojH)
        
        initial_angles = [70, 70]

        power_minimization = minimize(objective_function, initial_angles, args=(QWPgen, HWPgen, meter1), method='Nelder-Mead', tol=0.1)
        optimal_angles = power_minimization.x
        print(f"Optimal angles: QWPL = {optimal_angles[0]}, HWPL = {optimal_angles[1]}")
        print(f"Minimum power: {power_minimization.fun}")
        print(f"powerH_R {read_power(meter1)}")
        print(f"powerV_R {read_power(meter2)}")

        # RUN THE CODE BELOW ONCE THE ANGLES HAVE BEEN FOUND TO TAKE THE QPT

        data_file = 'C:\\Users\\USER\\results.csv' # WRITE TO A CSV TO INPUT THE CODE INTO MATHEMATICA 

        # This function takes the 36 measurements needed for the process tomogrraphy and stores them in results as a 6*6 array
        results = process_tomography(QWPgen, HWPgen, QWPproj, HWPproj, meter1, QWPgen_positions, QWPproj_positions)

        # Chi matrix test bench
        # eff_z = 10
        # values = [
        #     [eff_z, eff_z, eff_z, eff_z, eff_z, eff_z],
        #     [eff_z, 100e3, 50e3, 50e3, 50e3, 50e3],
        #     [eff_z, 50e3, 25e3, 25e3, 25e3, 25e3],
        #     [eff_z, 50e3, 25e3, 25e3, 25e3, 25e3],
        #     [eff_z, 50e3, 25e3, 25e3, 25e3, 25e3],
        #     [eff_z, 50e3, 25e3, 25e3, 25e3, 25e3]
        # ]

        # results = np.array(values)

        # -- Writing data to csv to then input into Mathematica as Python code doesn't seem to be working --
        write_results_to_csv(results, data_file)

        print("resulting matrix")
        print(results)
        print(f"results shape is {results.shape}")

        # -- IF PYTHON QPT CODE WORKS AGAIN, USE THE TEMPLATE BELOW --

        # # Run process tomography
        # chi_matrix = ProcessTomography(results)

        # # Print the resulting chi matrix
        # print("Chi Matrix:")
        # print(f"shape is {chi_matrix.shape}")
        # print(chi_matrix)

        # print("input purities")
        # input_p = input_purities(results)
        # print(input_p)

        # print("output purities")
        # output_p = output_purities(results, chi_matrix)
        # print(output_p)

        # # Create the figure and 3D axis
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # # Get the dimensions of the matrix
        # num_rows, num_cols = chi_matrix.shape

        # # Create a grid of positions for the bars
        # x, y = np.meshgrid(np.arange(num_cols), np.arange(num_rows))

        # # Flatten the x and y arrays and repeat the values for the bars
        # x = x.flatten()
        # y = y.flatten()
        # z = np.zeros_like(x)

        # # Flatten the chi_matrix values
        # values = chi_matrix.flatten()

        # # Define the width, depth, and height of the bars
        # dx = dy = 0.5
        # dz = values

        # # Create the bars
        # ax.bar3d(x, y, z, dx, dy, dz, shade=True)

        # # Set the labels and title
        # ax.set_xlabel('Sigma')
        # ax.set_ylabel('Sigma')
        # ax.set_zlabel('Value')
        # ax.set_xticks(np.arange(num_cols))
        # ax.set_yticks(np.arange(num_rows))
        # ax.set_xticklabels(['σ0', 'σ1', 'σ2', 'σ3'])
        # ax.set_yticklabels(['σ0', 'σ1', 'σ2', 'σ3'])
        # ax.set_title('Chi Matrix Visualization')

        # # Stop the timer
        end_time = time.time()

        # Calculate and print the time it takes for the QPT measurements
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")

        # # Show the plot
        # plt.show()


    finally:
        # Close devices and meter
        QWPgen.close_device()
        HWPgen.close_device()
        QWPproj.close_device()
        HWPproj.close_device()
        meter1.close() 
        meter2.close()

if __name__ == "__main__":
    main()
