from .walker import BanditWalker
from .mujoco_xml import BanditMujocoXmlEnv
from gym.envs.registration import register


class RoboschoolForwardWalkerMujocoXML(BanditWalker, BanditMujocoXmlEnv):
    def __init__(self, fn, robot_name, action_dim, obs_dim, power):
        BanditMujocoXmlEnv.__init__(self, fn, robot_name, action_dim, obs_dim)
        BanditWalker.__init__(self, power)


class RoboschoolAnt(RoboschoolForwardWalkerMujocoXML):
    foot_list = ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']

    def __init__(self):
        RoboschoolForwardWalkerMujocoXML.__init__(self, 'ant_mlsh.xml', "torso", action_dim=8, obs_dim=28, power=2.5)

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.26 else -1 # 0.25 is central sphere rad, die if it scrapes the ground


register(
    id="myAnt-v1",
    entry_point='pybullet_ant_bandit:RoboschoolAnt',
    max_episode_steps=1000,
    reward_threshold=950.0,
    tags={ "pg_complexity": 1*1000000 },
)
