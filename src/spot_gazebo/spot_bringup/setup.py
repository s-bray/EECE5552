from setuptools import setup
from glob import glob
import os

package_name = 'spot_bringup'

setup(
    name=package_name,
    version='0.0.1',
    packages=[],  # ← no python module directory to install
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Spot robot bringup (launch + config)',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={'console_scripts': []},
)

