from matplotlib import animation
import matplotlib.pyplot as plt
import gym

def save_frames_as_gif(frames, path='./', filename='output.gif'):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

"""
env = gym.make('LunarLander-v2')
frames = []
for t in range(1000):
    frames.append(env.render(mode='rgb_array'))
    action = env.action_space.sample()
    _,_,done, _ = env.step(action)
    if done:
        break
env.close()
save_frames_as_gif(frames)
"""