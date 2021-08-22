from gym.envs.registration import register

register(
    id='test_env-v0',
    entry_point='gym_test_env.envs:Test_envEnv',
)
register(
    id='test_env-extrahard-v0',
    entry_point='gym_test_env.envs:Test_envExtraHardEnv',
)