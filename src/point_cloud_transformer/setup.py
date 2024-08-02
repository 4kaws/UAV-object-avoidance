from setuptools import setup

package_name = 'point_cloud_transformer'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Point Cloud Transformer Node',
    license='License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'point_cloud_transformer_node = point_cloud_transformer.point_cloud_transformer_node:main'
        ],
    },
)

