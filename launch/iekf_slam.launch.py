import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

try:
    import yaml
except ImportError:
    yaml = None


def _to_env_str(value):
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value)


def _load_openmp_defaults(config_file):
    defaults = {
        "omp_num_threads": "4",
        "omp_dynamic": "FALSE",
        "omp_proc_bind": "TRUE",
        "omp_places": "cores",
    }
    if yaml is None:
        return defaults

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        params = cfg.get("iekf_slam_node", {}).get("ros__parameters", {})
        defaults["omp_num_threads"] = _to_env_str(
            params.get("runtime.omp_num_threads", defaults["omp_num_threads"])
        )
        defaults["omp_dynamic"] = _to_env_str(
            params.get("runtime.omp_dynamic", defaults["omp_dynamic"])
        )
        defaults["omp_proc_bind"] = _to_env_str(
            params.get("runtime.omp_proc_bind", defaults["omp_proc_bind"])
        )
        defaults["omp_places"] = _to_env_str(
            params.get("runtime.omp_places", defaults["omp_places"])
        )
    except Exception:
        # Keep launch robust even if YAML parsing fails.
        return defaults
    return defaults


def generate_launch_description():
    pkg_share = get_package_share_directory("iekf_lio")
    config_file = os.path.join(pkg_share, "config", "config.yaml")
    rviz_file = os.path.join(pkg_share, "rviz", "iekf_slam.rviz")
    omp_defaults = _load_openmp_defaults(config_file)

    omp_num_threads = LaunchConfiguration("omp_num_threads")
    omp_dynamic = LaunchConfiguration("omp_dynamic")
    omp_proc_bind = LaunchConfiguration("omp_proc_bind")
    omp_places = LaunchConfiguration("omp_places")

    iekf_node = Node(
        package="iekf_lio",
        executable="iekf_slam_node",
        name="iekf_slam_node",
        output="screen",
        parameters=[config_file],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_file],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "omp_num_threads",
                default_value=omp_defaults["omp_num_threads"],
                description="OpenMP thread count for heavy parallel stages",
            ),
            DeclareLaunchArgument(
                "omp_dynamic",
                default_value=omp_defaults["omp_dynamic"],
                description="Enable/disable OpenMP dynamic threading",
            ),
            DeclareLaunchArgument(
                "omp_proc_bind",
                default_value=omp_defaults["omp_proc_bind"],
                description="Bind OpenMP threads to processing units",
            ),
            DeclareLaunchArgument(
                "omp_places",
                default_value=omp_defaults["omp_places"],
                description="OpenMP thread placement policy",
            ),
            SetEnvironmentVariable("OMP_NUM_THREADS", omp_num_threads),
            SetEnvironmentVariable("OMP_DYNAMIC", omp_dynamic),
            SetEnvironmentVariable("OMP_PROC_BIND", omp_proc_bind),
            SetEnvironmentVariable("OMP_PLACES", omp_places),
            iekf_node,
            rviz_node,
        ]
    )
