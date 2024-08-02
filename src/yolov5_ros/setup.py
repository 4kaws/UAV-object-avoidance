from setuptools import setup
import os

package_name = 'yolov5_ros'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/yolov5_launch.py'])
    ],

    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Andrei Necula',  # Replace 'your_name' with your actual name
    maintainer_email='andrei.n53@yahoo.com',  # Replace with your actual email
    description='YOLOv5 ROS2 Object Detection Node',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_detector = yolov5_ros.object_detector:main'
        ],
    },
)
