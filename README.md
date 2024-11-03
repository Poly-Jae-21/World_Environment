# World_Environment
World Environment is an environment for developing reinforcement learning algorithm-based optimal urban planning of
electric vehicle charging networks by providing an API to communicate between learning algorithms and environments.

## Implementation
To implement the base World_Environment, we embody it on the Gymnasium (gym) interface, is pythonic of representing 
custom environment, and has compatibility wrapper through various functions:

```
import gymnasium as gym
# Initialize the environment 
env = gym.make("Chicago_world-v1", render_mode = "human")

# Reset the environment to generate the first observation
observation, _ = env.reset(seed=42)
```
