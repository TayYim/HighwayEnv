import numpy as np
from highway_env.vehicle.behavior import IDMVehicle
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle


class FrontCutInEnv(AbstractEnv):

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    # "features": ["x", "y", "vx", "vy", "heading"],
                    "normalize": False,
                    "absolute": True,
                    "vehicles_count": 2,
                },
                "real_time_rendering": True,
                "simulation_frequency": 15,
                "policy_frequency": 1,
                "absolute_v": 20,
                "relative_p": 20,
                "relative_v": -5,
            }
        )
        return config

    def _reward(self, action: int) -> float:
        """
        在这个场景中，没有外界指令对车辆进行控制，所以实际上没有用到reward函数。
        但是observation space和reward function必须在环境中被定义, 因此直接设置reward为常数.
        """
        return 0

    def _is_terminated(self) -> bool:
        """当车辆发生碰撞，episode结束 ."""
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        return False

    def _cost(self, action: int) -> float:
        return float(self.vehicle.crashed)

    def _reset(self) -> np.ndarray:
        """初始化环境"""
        self._make_road()
        self._make_vehicles()

    def _make_road(self, length=1000):
        """
        生成一条包括两个车道的路.

        :返回: the road
        """
        net = RoadNetwork()

        # 设定车道，路的两个端点分别是a和b
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [0, 0],
                [length, 0],
                line_types=(LineType.CONTINUOUS_LINE, LineType.STRIPED),
            ),
        )
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [0, StraightLane.DEFAULT_WIDTH],
                [length, StraightLane.DEFAULT_WIDTH],
                line_types=(LineType.NONE, LineType.CONTINUOUS_LINE),
            ),
        )

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _make_vehicles(self) -> None:
        """
        在生成的路上填充车辆, 包括ego车辆和有挑战行为的车辆.
        """
        road = self.road
        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        # 设置ego车辆为IDM模型控制.
        ego_vehicle = vehicles_type(
            road,
            position=road.network.get_lane(("a", "b", 1)).position(40, 0),
            heading=0,
            speed=self.config["absolute_v"],
            enable_lane_change=True,
        )
        ego_vehicle.set_ego()
        ego_vehicle.color = (0, 255, 0)  # green
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        # 设置由IDM模型控制的前车切入车辆.
        v = vehicles_type(
            road,
            position=road.network.get_lane(("a", "b", 0)).position(
                40
                + self.config["relative_p"]
                + vehicles_type.LENGTH * np.sign(self.config["relative_p"]),
                0,
            ),
            heading=0,
            speed=self.config["absolute_v"] + self.config["relative_v"],
            enable_lane_change=False,
        )
        road.vehicles.append(v)

    def step(self, action):
        # 在steps==3时,进行变道.
        if self.time == 3:
            self.road.vehicles[1].target_lane_index = ("a", "b", 1)

        # use super
        return super().step(action)


class FrontCutInWithTwoNPCEnv(AbstractEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "features": ["presence", "x", "y", "vx", "vy"],
                    "normalize": False,
                    "absolute": True,
                    "vehicles_count": 3,
                },
                "real_time_rendering": True,
                "simulation_frequency": 15,
                "policy_frequency": 1,
                "lane_change_time": 3,
                "absolute_v": 0,
                "relative_p_1": 0,
                "relative_v_1": 0,
                "relative_p_2": 0,
                "relative_v_2": 0,
            }
        )
        return config

    def _reward(self, action: int) -> float:
        """
        在这个场景中,没有外界指令对车辆进行控制,所以实际上没有用到reward函数。
        但是observation space和reward function必须在环境中被定义, 因此直接设置reward为常数.
        """
        return 0

    def _is_terminated(self) -> bool:
        """当车辆发生碰撞,episode结束 ."""
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        return False

    def _cost(self, action: int) -> float:
        return float(self.vehicle.crashed)

    def _reset(self) -> np.ndarray:
        """初始化环境"""
        self._make_road()
        self._make_vehicles()

    def _make_road(self, length=1000):
        """
        生成一条包括两个车道的路.

        :返回: the road
        """
        net = RoadNetwork()

        # 设定车道,路的两个端点分别是a和b
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [0, 0],
                [length, 0],
                line_types=(LineType.CONTINUOUS_LINE, LineType.STRIPED),
            ),
        )
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [0, StraightLane.DEFAULT_WIDTH],
                [length, StraightLane.DEFAULT_WIDTH],
                line_types=(LineType.NONE, LineType.CONTINUOUS_LINE),
            ),
        )

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _make_vehicles(self) -> None:
        """
        在生成的路上填充车辆, 包括ego车辆和有挑战行为的车辆.
        """
        road = self.road
        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        starting_position_x = 40

        # 设置ego车辆为IDM模型控制.
        ego_vehicle = vehicles_type(
            road,
            position=road.network.get_lane(("a", "b", 1)).position(
                starting_position_x, 0
            ),
            heading=0,
            speed=self.config["absolute_v"],
            enable_lane_change=True,
        )
        ego_vehicle.set_ego()
        ego_vehicle.color = (0, 255, 0)  # green
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        # 设置由IDM模型控制的前车切入车辆.
        v = vehicles_type(
            road,
            position=road.network.get_lane(("a", "b", 0)).position(
                starting_position_x
                + self.config["relative_p_1"]
                + vehicles_type.LENGTH * np.sign(self.config["relative_p_1"]),
                0,
            ),
            heading=0,
            speed=self.config["absolute_v"] + self.config["relative_v_1"],
            enable_lane_change=False,
        )
        road.vehicles.append(v)
        self.lc_vehicle = v

        v2 = vehicles_type(
            road,
            position=road.network.get_lane(("a", "b", 0)).position(
                starting_position_x
                + self.config["relative_p_2"]
                + vehicles_type.LENGTH * np.sign(self.config["relative_p_2"]),
                0,
            ),
            heading=0,
            speed=self.config["absolute_v"] + self.config["relative_v_2"],
            enable_lane_change=False,
        )
        road.vehicles.append(v2)

    def step(self, action):
        if self.time == self.config["lane_change_time"]:
            self.lc_vehicle.target_lane_index = ("a", "b", 1)

        # use super
        return super().step(action)

