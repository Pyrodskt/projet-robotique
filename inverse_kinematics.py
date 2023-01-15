import ikpy.chain as ik
import warnings
import time
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D

def compute_inverse_kinematics(position_matrix, orientation_matrix, calibration=False, display=False):
  if(calibration):
    warnings.filterwarnings("ignore")
    chain = ik.Chain.from_urdf_file("urdf_files/calibration.urdf")
    warnings.filterwarnings("default")
  else:
    warnings.filterwarnings("ignore")
    chain = ik.Chain.from_urdf_file("urdf_files/normal.urdf")
    warnings.filterwarnings("default")

  print("Begin inverse kinematics compute...")
  start = time.time()
  inverse_kinematics = chain.inverse_kinematics(position_matrix, orientation_matrix)
  stop = time.time()
  print("Ok ! Took "+str(stop-start)+"s")

  if(display):
    ax = matplotlib.pyplot.figure().add_subplot(111, projection='3d')
    chain.plot(inverse_kinematics, ax)
    matplotlib.pyplot.show()
  return inverse_kinematics
