import os

import numpy as np
import pandas as pd

from array2gif import write_gif
from PIL import Image

from utils.simple_environment import SimpleEnvironment


def generate_data(generation_params):

    width, height = generation_params['width'], generation_params['height']
    agent_radius = generation_params['agent_diameter']
    num_samples = generation_params['num_samples']
    occlusion_rate = generation_params['occlusion_rate']
    data_name = generation_params['dataset_name']

    # Init. Environment and Add agent
    environment = SimpleEnvironment(width, height, occlusion_rate)
    environment.add_agent((0, 0), 16, shape='circle')  # obstacle
    environment.add_agent((40, 0), 8, shape='circle', color=(1.0, 0.0, 0.0))  # objective point

    # Create random trajectory
    samples_x = np.random.randint(low=-width//2, high=width//2, size=num_samples)
    samples_y = np.random.randint(low=-height//2, high=height//2, size=num_samples)
    trajectory = [coord for coord in zip(samples_x, samples_y)]

    # Get trajectory images
    trajectory_images = environment.draw_trajectory(trajectory, agent_radius)

    # Save to excel
    if not os.path.exists(data_name):
        os.makedirs(data_name)

    entries = []
    for i, (img, coord) in enumerate(zip(trajectory_images, trajectory)):

        img_name = '{}/img{}.png'.format(data_name, i)
        im = Image.fromarray((img * 255).astype(np.uint8))
        im.save(img_name)

        entries.append([img_name, coord])

    df = pd.DataFrame(entries, columns=['Path', 'Agent Coordinate'])
    df.to_excel(data_name + '.xlsx')

    # Save as GIF
    # build_gif(trajectory_images, data_name + '.gif')


def build_gif(sources, name, fps=6):

    joined = sources
    joined = 255.0 * np.array(joined).squeeze()

    write_gif(joined, name, fps=fps)


if __name__ == '__main__':

    gen_params = dict(
        width=100,  #500,
        height=60,  #300,
        agent_diameter=4,  #10,
        occlusion_rate=0.0,   #1.0,
        num_samples=10000,
        dataset_name='validation'
    )

    generate_data(gen_params)
