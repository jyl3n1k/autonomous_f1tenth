import os
from glob import glob

from setuptools import find_packages, setup


package_name = 'turtlebot3_vision_controller'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py'),
        ),
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*.yaml'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jylen',
    maintainer_email='jylen@todo.todo',
    description=(
        'OpenCV corridor-following controller for the simulated '
        'TurtleBot3 Burger.'
    ),
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision_controller = '
            'turtlebot3_vision_controller.controller:main',
        ],
    },
)
