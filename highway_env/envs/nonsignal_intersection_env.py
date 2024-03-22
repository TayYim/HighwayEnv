from typing import Dict, Tuple
import sys
import os
import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv, MultiAgentWrapper
from highway_env.road.lane import LineType, StraightLane, CircularLane, AbstractLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.road.road import Road, RoadNetwork


class OppositeVehicleTakingPriority(AbstractEnv):

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "action": {
                    "type": "DiscreteMetaAction",
                    "longitudinal": True,
                    "lateral": False,
                    "target_speeds": [0, 4.5, 9],
                },
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 5,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "normalize": True,
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-20, 20],
                        "vy": [-20, 20],
                    },
                    "absolute": False,
                    # observe_intentions enable the computation of the intended destination of each observed vehicle
                    # (if they have a planned route).
                    # When enabled, the cos_d, sin_d fields also have to be added to the observation's list of features
                    "observe_intentions": False,
                },
                "vehicles_count": 5,
                "search": True,
                "speed_limit": 20,
                "v_ego": 5,
                "r_ego": 20,
                "v_1": 5,
                "r_1": 20,
                "lane_width": 3.7,
                "access_length": 50,
                "duration": 20,  # [s]
                "simulation_frequency": 15,
                "policy_frequency": 1,
                "destination": "o1",
                "controlled_vehicles": 1,
                "initial_vehicle_count": 2,
                "screen_width": 700,
                "screen_height": 700,
                # centering_position是在渲染窗口里车辆的[水平, 垂直]位置。
                # 举例:
                # [0, 0] --> 左上
                # [0.5, 0.5] --> 中间
                # [1, 1] --> 右下
                "centering_position": [0.5, 0.5],
                "scaling": 5.5 * 1.3,
                "ego_start_lane_index": ("o0", "ir0", 0),
                "ego_plan_to": "o2",
                "npc_start_lane_index": ("o1", "ir1", 0),
                "npc_plan_to": "o3",
            }
        )
        return config

    def _reward(self, action: int) -> float:
        return 0

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        return 0

    def _cost(self, action: int) -> float:
        return float(self.vehicle.crashed)

    def _is_terminated(self) -> bool:
        has_crash = any(vehicle.crashed for vehicle in self.controlled_vehicles)
        any_arrived = any(
            self.has_arrived(vehicle) for vehicle in self.controlled_vehicles
        )
        timeout = self.time >= self.config["duration"] * self.config["policy_frequency"]
        return has_crash or any_arrived or timeout

    def _is_truncated(self) -> bool:
        return False

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return (
            vehicle.crashed
            or self.time >= self.config["duration"] * self.config["policy_frequency"]
            or self.has_arrived(vehicle)
        )

    def _info(self, obs: np.ndarray, action: int) -> dict:
        info = super()._info(obs, action)
        info["agents_rewards"] = tuple(
            self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles
        )
        info["agents_dones"] = tuple(
            self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles
        )
        return info

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles(self.config["initial_vehicle_count"])

    def step(self, action):
        return super().step(action)

    def _make_road(self) -> None:
        """
        Make an 4-way intersection.

        The horizontal road has the right of way. More precisely, the levels of priority are:
            - 3 for horizontal straight lanes and right-turns
            - 1 for vertical straight lanes and right-turns
            - 2 for horizontal left-turns
            - 0 for vertical left-turns

        路网节点编码:
        (o:外 | i:内 + [r:右, l:左]) + (0:south | 1:west | 2:north | 3:east)

        :返回: the intersection road
        """
        # lane_width = AbstractLane.DEFAULT_WIDTH # 4
        lane_width = self.config["lane_width"]
        right_turn_radius = lane_width + 5  # 9[m}
        left_turn_radius = right_turn_radius + lane_width  # 13[m}
        outer_distance = right_turn_radius + lane_width / 2  #
        access_length = self.config["access_length"]  # [m]

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        for corner in range(4):
            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            priority = 3 if is_horizontal else 1
            rotation = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            # Incoming
            start = rotation @ np.array(
                [lane_width / 2, access_length + outer_distance]
            )
            end = rotation @ np.array([lane_width / 2, outer_distance])
            net.add_lane(
                "o" + str(corner),
                "ir" + str(corner),
                StraightLane(
                    start,
                    end,
                    line_types=[s, c],
                    priority=priority,
                    speed_limit=self.config["speed_limit"],
                ),
            )
            # Right turn
            r_center = rotation @ (np.array([outer_distance, outer_distance]))
            net.add_lane(
                "ir" + str(corner),
                "il" + str((corner - 1) % 4),
                CircularLane(
                    r_center,
                    right_turn_radius,
                    angle + np.radians(180),
                    angle + np.radians(270),
                    line_types=[n, c],
                    priority=priority,
                    speed_limit=self.config["speed_limit"],
                ),
            )
            # Left turn
            l_center = rotation @ (
                np.array(
                    [
                        -left_turn_radius + lane_width / 2,
                        left_turn_radius - lane_width / 2,
                    ]
                )
            )
            net.add_lane(
                "ir" + str(corner),
                "il" + str((corner + 1) % 4),
                CircularLane(
                    l_center,
                    left_turn_radius,
                    angle + np.radians(0),
                    angle + np.radians(-90),
                    clockwise=False,
                    line_types=[n, n],
                    priority=priority - 1,
                    speed_limit=self.config["speed_limit"],
                ),
            )
            # # Straight
            start = rotation @ np.array([lane_width / 2, outer_distance])
            end = rotation @ np.array([lane_width / 2, -outer_distance])
            net.add_lane(
                "ir" + str(corner),
                "il" + str((corner + 2) % 4),
                StraightLane(
                    start,
                    end,
                    line_types=[n, n],
                    priority=priority,
                    speed_limit=self.config["speed_limit"],
                ),
            )
            # # Exit
            start = rotation @ np.flip(
                [lane_width / 2, access_length + outer_distance], axis=0
            )
            end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
            net.add_lane(
                "il" + str((corner - 1) % 4),
                "o" + str((corner - 1) % 4),
                StraightLane(
                    end,
                    start,
                    line_types=[n, c],
                    priority=priority,
                    speed_limit=self.config["speed_limit"],
                ),
            )

        road = RegulatedRoad(network=net, np_random=self.np_random)
        self.road = road

    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane

        :return: the ego-vehicle
        """
        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        # predict_vehicle_type = utils.class_from_path(self.config["predict_IDM_vehicle"])
        self.controlled_vehicles = []

        r_ego, r_1 = self.compute_initial_param()
        # 主车
        ego_vehicle = vehicle_type.make_on_lane(
            self.road,
            self.config["ego_start_lane_index"],
            longitudinal=r_ego,
            speed=self.config["v_ego"],
        )
        ego_vehicle.REVERSE = False
        ego_vehicle.PREDICT_DIS = 2
        ego_vehicle.MARGIN = 5
        # 设置主车的终点
        try:
            ego_vehicle.plan_route_to(self.config["ego_plan_to"])
            ego_vehicle.target_speed = self.config["speed_limit"]
        except AttributeError as e:
            print(f"Error when init ego:{e}")

        ego_vehicle.is_ego = True
        self.vehicle = ego_vehicle
        self.road.vehicles.append(ego_vehicle)
        self.controlled_vehicles.append(ego_vehicle)

        for v in self.road.vehicles:  # Prevent early collisions
            if (
                v is not ego_vehicle
                and np.linalg.norm(v.position - ego_vehicle.position) < 20
            ):
                self.road.vehicles.remove(v)

        # # 设置背景车辆
        vehicle2 = vehicle_type.make_on_lane(
            self.road,
            self.config["npc_start_lane_index"],
            longitudinal=r_1,
            speed=self.config["v_1"],
        )
        vehicle2.REVERSE = False
        vehicle2.plan_route_to(self.config["npc_plan_to"])
        vehicle2.target_speed = self.config["speed_limit"]
        self.road.vehicles.append(vehicle2)

    # 相对路口中心的距离
    def compute_initial_param(self):
        r_1 = (
            self.config["lane_width"] * 2
            + 5
            + self.config["access_length"]
            - self.config["r_1"]
            - 5 / 2
        )  # 5 是车长
        r_ego = (
            self.config["lane_width"] * 2
            + 5
            + self.config["access_length"]
            - self.config["r_ego"]
            - 5 / 2
        )
        return r_ego, r_1

    def _clear_vehicles(self) -> None:
        is_leaving = (
            lambda vehicle: "il" in vehicle.lane_index[0]
            and "o" in vehicle.lane_index[1]
            and vehicle.lane.local_coordinates(vehicle.position)[0]
            >= vehicle.lane.length - 4 * vehicle.LENGTH
        )
        self.road.vehicles = [
            vehicle
            for vehicle in self.road.vehicles
            if vehicle in self.controlled_vehicles
            or not (is_leaving(vehicle) or vehicle.route is None)
        ]

    def has_arrived(self, vehicle: Vehicle, exit_distance: float = 5) -> bool:
        vehicle_route = vehicle.route
        if vehicle_route is None or len(vehicle_route) == 0:
            return False
        if (
            vehicle_route[-1][0] in vehicle.lane_index[0]
            and vehicle_route[-1][1] in vehicle.lane_index[1]
        ):
            if vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance:
                return True
            else:
                return False
        else:
            return False


class NonSignalizedJunctionLeftTurn(OppositeVehicleTakingPriority):

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "ego_start_lane_index": ("o0", "ir0", 0),
                "ego_plan_to": "o1",
                "npc_start_lane_index": ("o2", "ir2", 0),
                "npc_plan_to": "o0",
            }
        )
        return config


class NonSignalizedJunctionRightTurn(OppositeVehicleTakingPriority):

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "ego_start_lane_index": ("o0", "ir0", 0),
                "ego_plan_to": "o3",
                "npc_start_lane_index": ("o1", "ir1", 0),
                "npc_plan_to": "o3",
            }
        )
        return config
