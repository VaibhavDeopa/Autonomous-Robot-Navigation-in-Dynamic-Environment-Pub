import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'nav_env_pkg'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.wbt')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'resource'), glob('resource/*.urdf') + glob('resource/*.yml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='uday',
    maintainer_email='uday@todo.todo',
    description='DRL Navigation for TurtleBot3',
    license='MIT',
    entry_points={'console_scripts': []},
)
