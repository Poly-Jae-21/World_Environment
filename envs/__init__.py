from gymnasium.envs.registration import register

register(
    id = 'UrbanEnvChicago-v1',
    entry_point='envs.multi_policies:ChicagoMultiPolicyMap',
)