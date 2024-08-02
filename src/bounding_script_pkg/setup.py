from setuptools import setup
import os
from glob import glob

package_name = 'bounding_script_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='andrei',
    maintainer_email='andrei.n53@yahoo.com',
    description='A ROS2 package for image processing with a bounding script',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_subscriber = bounding_script_pkg.image_subscriber:main',
        ],
    },
)


