import numpy as np

import matplotlib.pyplot as plt


class SimpleEnvironment:

    def __init__(self, width, height, occlusion_rate=0.0):
        assert width > 0 and height > 0, 'Width and Height must be greater than zero.'

        self._width = width
        self._height = height
        self._field = np.zeros((self._height, self._width, 3))
        self._obstacles = []

        self._occlusion_rate = occlusion_rate

    def get_field_copy(self):

        return self._field.copy()

    def add_agent(self, coord, diameter=4, shape='square', color=(1.0, 1.0, 1.0)):

        diameter = abs(diameter)
        if diameter % 2 != 0:
            diameter += 1

        num_channels = 3
        agent = np.zeros((diameter, diameter, num_channels))

        if shape == 'circle':
            # Set Color
            inside = lambda x, y: (x - diameter // 2) ** 2 + (y - diameter // 2) ** 2 < (diameter // 2) ** 2
            for i in range(diameter):
                for j in range(diameter):
                    if inside(i, j):
                        agent[j, i, :] = color
        else:  # shape == 'square'
            # Set color
            for i in range(num_channels):
                agent[:, :, i] = np.full((diameter, diameter), color[i])

        self._draw(agent, coord)

    def draw_trajectory(self, trajectory, diameter=4, shape='square', color=(1.0, 1.0, 1.0), ignore_occlusion=False):

        diameter = abs(diameter)
        if diameter % 2 != 0:
            diameter += 1

        num_channels = 3
        agent = np.zeros((diameter, diameter, num_channels))

        if shape == 'circle':
            # Set Color
            inside = lambda x, y: (x - diameter // 2) ** 2 + (y - diameter // 2) ** 2 < (diameter // 2) ** 2
            for i in range(diameter):
                for j in range(diameter):
                    if inside(i, j):
                        agent[j, i, :] = color
        else:  # shape == 'square'
            # Set color
            for i in range(num_channels):
                agent[:, :, i] = np.full((diameter, diameter), color[i])

        drawing = []
        for coord in trajectory:
            past, y_coord, x_coord = self._draw(agent, coord)

            copy = self.get_field_copy()

            # Add occlusion with probability `self._occlusion_rate`
            if (not ignore_occlusion) and (np.random.rand() < self._occlusion_rate):
                occlusion = self._get_occlusion_element()
                for x in range(copy.shape[1]):
                    for y in range(copy.shape[0]):
                        pixel_val = copy[y, x, :] + occlusion[y, x, :]
                        if np.any(pixel_val > 1):
                            pixel_val = 0.5 * copy[y, x, :] + occlusion[y, x, :]
                        copy[y, x, :] = pixel_val

            drawing.append(copy)
            self._field[y_coord[0]:y_coord[1], x_coord[0]:x_coord[1], :] = past

        return drawing

    # Private methods
    def _draw(self, element, coord):

        w, h, _ = element.shape
        assert w < self._width and h < self._width, 'The element must fit on the instance\'s field.'
        w = w // 2
        h = h // 2

        x, y = self._cartesian_to_index(coord)
        x_lhs, y_up = max(x - w, 0), max(y - h, 0)
        x_rhs, y_bottom = min(x + w, self._width), min(y + h, self._height)

        # Truncate element to fit inside board
        element = element.copy()
        if x - w < 0:
            element = element[:, w - x:, :]

        if x + w > self._width:
            element = element[:, :x_rhs - x_lhs, :]

        if y - h < 0:
            element = element[h - y:, :, :]

        if y + h > self._height:
            element = element[:y_bottom - y_up, :, :]

        past = self._field[y_up:y_bottom, x_lhs:x_rhs, :].copy()
        if past.shape == element.shape:
            self._field[y_up:y_bottom, x_lhs:x_rhs, :] = element
        else:
            print('Could not perform assignment')
            print(y_up, y_bottom, x_lhs, x_rhs)

        return past, (y_up, y_bottom), (x_lhs, x_rhs)

    def _get_occlusion_element(self):
        # Simulate cloud as occlusion element

        def rand_coord(w, h):
            coord = (np.random.randint(low=-w//2, high=w//2), np.random.randint(low=-h//2, high=h//2))
            return self._cartesian_to_index(coord)

        inside = lambda x, y: ((x - x0) - diameter // 2) ** 2 + ((y - y0) - diameter // 2) ** 2 < (diameter // 2) ** 2

        occlusion = np.zeros((self._height, self._width, 3))
        cloud_center = rand_coord(self._width, self._height)
        alpha = np.random.uniform(low=0.3, high=0.7, size=1)
        num_bubbles = np.random.randint(low=1, high=5, size=1)[0]
        # print('NUM BUBBLES =', num_bubbles)
        for _ in range(num_bubbles):
            # Generate bubble's diameter and deviation from cloud_center
            diameter = np.random.uniform(low=3, high=30, size=1)
            deviation = np.random.uniform(low=-10, high=10, size=1)

            color = (0, 0, alpha)
            x0, y0 = cloud_center[0] + deviation, cloud_center[1] + deviation
            # print('Coords:', x0, y0, diameter)

            for i in range(occlusion.shape[0]):
                for j in range(occlusion.shape[1]):
                    if inside(j, i):
                        occlusion[i, j, :] = color

        return occlusion

    def _cartesian_to_index(self, coord):

        x, y = coord
        x = x + self._width // 2
        y = -y + self._height // 2

        return round(x), round(y)


if __name__ == '__main__':
    # agent = np.zeros((4, 4, 3))
    # agent[:2, :2, 0] = np.ones((2, 2))
    # agent[2:, :2, 1] = np.ones((2, 2))
    # agent[:2, 2:, 2] = np.ones((2, 2))
    # agent[2:, 2:, :] = np.ones((2, 2, 3))
    # agent = np.ones((100, 100, 3))

    # env = SimpleEnvironment(600, 600)
    # env.draw(agent, (0, 0))
    # env.add_agent((0, 0), radius=20, color=(1, 1, 0))
    # env.add_agent((100, -50), radius=100, color=(0, 1, 1), shape='circle')
    # env.add_agent((-180, 100), radius=100, color=(0.3, 0.12, 0.4))

    environment = SimpleEnvironment(100, 60, occlusion_rate=0.5)
    environment.add_agent((0, 0), 16, shape='circle')  # obstacle
    environment.add_agent((40, 0), 8, shape='circle', color=(1.0, 0.0, 0.0))  # objective point

    plt.imshow(environment.get_field_copy())
    plt.show()
