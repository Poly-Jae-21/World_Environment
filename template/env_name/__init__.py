from gymnasium.envs.registration import register

register(
    id = 'UrbanEnvChicago-v1',
    entry_point='env_name.envs.multi_policies:ChicagoMultiPolicyMap',
)