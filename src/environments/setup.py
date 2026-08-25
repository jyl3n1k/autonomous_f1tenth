from setuptools import setup
import os
from glob import glob

package_name = 'environments'

folders = glob('map_info/*')
map_infos = []
for folder in folders:
    try:
        map_infos.append((os.path.join('share', package_name, 'map_info'), glob(f"{folder}/*")))
    except:
        print("Not working")

model_files = []
for root, _, files in os.walk('models'):
    if files:
        model_files.append((
            os.path.join('share', package_name, root),
            [os.path.join(root, filename) for filename in files]
        ))

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, 
              f'{package_name}.autoencoders'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'sdf'), glob('sdf/*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*')),
        (os.path.join('share', 'f1tenth', 'f1tenth_description', 'meshes'), glob('meshes/*')),
        *model_files,
        *map_infos,
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='retinfai',
    maintainer_email='aferetipama@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'CarTrackReset = environments.CarTrackReset:main',
            'CarBeatReset = environments.CarBeatReset:main',
            'CarRaceReset = environments.CarRaceReset:main',
            'CarOvertakeReset = environments.CarOvertakeReset:main',
            'TwoCarReset = environments.TwoCarReset:main',
            'SteppingService = environments.SteppingService:main',
            'LidarLogger = environments.lidar_logger:main',
            'MultiAgentReset = environments.MultiAgentReset:main',
            'F1TenthReset = environments.F1TenthReset:main'
        ],
    },
)
