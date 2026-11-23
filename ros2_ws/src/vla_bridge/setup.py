from setuptools import find_packages, setup

setup(
    name="vla_bridge",
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/vla_bridge"]),
        ("share/vla_bridge", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    entry_points={"console_scripts": ["vla_node = vla_bridge.vla_node:main"]},
)
