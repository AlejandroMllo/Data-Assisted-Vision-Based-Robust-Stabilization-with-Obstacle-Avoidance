from copy import deepcopy

from utils.generate_data import build_gif
from utils.simple_environment import SimpleEnvironment

from tensorflow import keras, make_tensor_proto, make_ndarray

import matplotlib.pyplot as plt
import numpy as np


def main(params):

    trajectory, init_x, init_y, diff = waypoint_tracking(params)

    # trajectory, init_x, init_y, diff = pursue_objective(params)
    # if params['use_learning']:
    #     drawing = trajectory
    #     print('no need for new drawing')
    # else:
    #     environment = create_environment(params)
    #     drawing = environment.draw_trajectory(trajectory)
    #
    # build_gif(trajectory, 'no_pred_init({:.1f}, {:.1f})_diff_{:.2f}.gif'.format(init_x, init_y, diff))


def pursue_objective(params):

    # Parameter retrieval
    obs_pos = params['obstacle_coord']
    obs_diameter = params['obstacle_diameter']
    obj_pos = params['objective_coord']
    stop_difference = params['stop_difference']
    use_learning = params['use_learning']
    preprocessing = params['preprocessing']

    # Constants
    obs_safe_distance = obs_diameter // 2

    # Convenience Functions
    def calculate_difference(pos):
        return np.sqrt((pos[0] - obj_pos[0]) ** 2 + (pos[1] - obj_pos[1]) ** 2)

    def get_prediction(pred):
        return list(make_ndarray(make_tensor_proto(pred))[0])

    # Set agent's initial position (coordinates)
    init_x = -40  # -28  # np.random.randint(low=-width//2, high=width//2)
    init_y = 0  # -11  # np.random.randint(low=-height//2, high=height//2)

    # Initialize the state at the agent's initial position
    state = []
    if use_learning:
        environment = create_environment(params)
        model = load_model(params)

        drawing = environment.draw_trajectory([[init_x, init_y]])
        state.append(drawing[0])

        # plt.imshow(state[-1])
        # plt.title('INIT: ({}, {})'.format(init_x, init_y))
        # plt.show()
    else:
        state.append([init_x, init_y])

    # Controller parameters
    i = 0
    a = 0.45
    b = 1 - a
    dt = 0.05

    x, y = init_x, init_y
    while calculate_difference([x, y]) > stop_difference:
        i += 1

        # Get Previous state
        if use_learning:
            prev_state = preprocessing(state[-1])
            predicted_state = model(prev_state)
            print('iteration', i)
            print('pred state', predicted_state)
            x_t, y_t = get_prediction(predicted_state)
        else:
            x_t, y_t = state[-1]

        # Controller
        mx = 1 if x_t > 1 else -1
        my = 1 if y_t > 1 else -1

        obj_x = (x_t - obj_pos[0])
        obs_x = (obs_diameter + obs_safe_distance - mx * (x_t - obs_pos[0]))

        obj_y = (y_t - obj_pos[1])
        obs_y = (obs_diameter + obs_safe_distance - my * (y_t - obs_pos[1]))

        ref = 1e-1
        if len(state) > 3:
            if use_learning:
                if np.allclose(state[-1], state[-3], rtol=ref):
                    a = a * 1.05
                    b = 1 - a
                    print('inside img state')
            else:
                if abs(state[-1][0] - state[-2][0]) < ref and abs(state[-1][1] == state[-2][1]) < ref:
                    a = a * 1.05
                    b = 1 - a
                    print('inside num state')

        ux = -a * obj_x + b * obs_x
        uy = -a * obj_y + b * obs_y

        x = x_t + dt * ux
        y = y_t + dt * uy

        if use_learning:
            print('new state ({}, {})'.format(x, y))
            drawing = environment.draw_trajectory([[x, y]])
            state.append(drawing[0])

            plt.imshow(state[-1])
            plt.title('(x, y) = ({:.2f}, {:.2f}) | pred = ({:.2f}, {:.2f})'.format(x, y, x_t, y_t))
            plt.show()
        else:
            state.append([x, y])

    print('Stopped in iteration {}'.format(i))
    diff = calculate_difference(state[-1])

    return state, init_x, init_y, diff


def waypoint_tracking(params):

    # Parameter retrieval
    width, height = params['env_dimension']
    obs_pos = params['obstacle_coord']
    obs_diameter = params['obstacle_diameter']
    objective_pos = params['objective_coord']
    stop_difference = params['stop_difference']
    preprocessing = params['preprocessing']

    # Convenience Functions
    def calculate_difference(params):

        pos = params['state'][-1]
        obj = params['obj_pos']

        return np.sqrt((pos[0] - obj[0]) ** 2 + (pos[1] - obj[1]) ** 2)

    def get_prediction(pred):
        return list(make_ndarray(make_tensor_proto(pred))[0])

    # Set agent's initial position (coordinates)
    init_x = -40    # np.random.randint(low=-width//2, high=width//2)
    init_y = 0      # np.random.randint(low=-height//2, high=height//2)

    # Initialize the state at the agent's initial position
    environment = create_environment(params)
    model = load_model(params)

    # Controller parameters
    i = 0
    reference_params = dict(
        a=0.3, b=1 - 0.2, dt=0.05,
        state=[[init_x, init_y]],
        obj_pos=objective_pos,
        obstacle_coord=obs_pos,
        obstacle_diameter=obs_diameter,
        color=(1.0, 1.0, 1.0)
    )
    tracker_params = deepcopy(reference_params)
    tracker_params['a'] = 1.0  #0.45
    tracker_params['b'] = 1 - tracker_params['a']
    tracker_params['color'] = (0.0, 1.0, 0.0)

    reference_params['obj_pos'] = objective_pos

    trajectory = []
    while calculate_difference(reference_params) > stop_difference:
        i += 1
        if i > 300:
            break

        d = tracker_params['state'][-1]
        drawing = environment.draw_trajectory([d], color=tracker_params['color'])[0]
        d2 = reference_params['state'][-1]
        drawing2 = environment.draw_trajectory([d2])[0]

        d_comb = np.clip(drawing + drawing2, 0.0, 1.0)
        trajectory.append(d_comb)

        # Get Previous state
        x_t, y_t = reference_params['state'][-1]

        ##################
        drawing = environment.draw_trajectory([[x_t, y_t]])  # Take picture of environment to predict objective's state.
        prev_state = preprocessing(drawing[0])
        predicted_state = model(prev_state)
        predicted_state = get_prediction(predicted_state)
        tracker_params['obj_pos'] = predicted_state
        ##################

        # Move Tracker
        tracker_params = controller(tracker_params)

        # Move Reference
        reference_params = controller(reference_params)

    print('Waypoint Tracking stopped in iteration {}'.format(i))
    diff = calculate_difference(reference_params)
    plot_trajectory_normal_controller(reference_params, tracker_params)

    return trajectory, init_x, init_y, diff


def controller(params):

    state = params['state']
    current_pos = state[-1]
    obj_pos = params['obj_pos']
    if isinstance(obj_pos[0], list):
        obj_pos = obj_pos[-1]

    obs_pos = params['obstacle_coord']
    obs_diameter = params['obstacle_diameter']
    obs_safe_distance = obs_diameter // 2

    # Controller
    mx = 1 if current_pos[0] > 1 else -1
    my = 1 if current_pos[1] > 1 else -1

    obj_x = (current_pos[0] - obj_pos[0])
    obs_x = (obs_diameter + obs_safe_distance - mx * (current_pos[0] - obs_pos[0]))

    obj_y = (current_pos[1] - obj_pos[1])
    obs_y = (obs_diameter + obs_safe_distance - my * (current_pos[1] - obs_pos[1]))

    ref = 1e-1
    if len(state) > 2:
        if abs(state[-1][0] - state[-2][0]) < ref and abs(state[-1][1] == state[-2][1]) < ref:
            params['a'] = params['a'] * 1.05
            params['b'] = 1 - params['a']
            print('change')

    a = params['a']
    b = params['b']
    dt = params['dt']

    ux = -a * obj_x + b * obs_x
    uy = -a * obj_y + b * obs_y

    x = current_pos[0] + dt * ux
    y = current_pos[1] + dt * uy

    params['state'].append([x, y])

    return params


def plot_trajectory_normal_controller(reference_params, tracker_params):

    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.grid(True, ls=':', lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    # Static elements
    obstacle = plt.Circle((0, 0), radius=8, ls='-', color='w')
    objective = plt.Circle((40, 0), radius=4, color='r')
    ax.add_patch(obstacle)
    ax.add_patch(objective)

    # Labels
    plt.suptitle('Agents\' trajectory')
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')

    # Plot config.
    ax.set_facecolor('k')
    ax.set_xlim(left=-50, right=50)
    ax.set_ylim(bottom=-30, top=30)
    plt.xticks(ticks=np.arange(-50, 50, 10))
    plt.yticks(ticks=np.arange(-30, 30, 10))

    x = [xi for xi, _ in reference_params['state']]
    y = [yi for _, yi in reference_params['state']]
    plt.plot(x, y, c='w', label='Leader')

    x = [xi for xi, _ in tracker_params['state']]
    y = [yi for _, yi in tracker_params['state']]
    plt.plot(x, y, c='g', label='Follower', ls='--')

    x0, y0 = reference_params['state'][0]
    ax.text(x0+1, y0-3, r'$({}, {})$'.format(x0, y0), fontsize=8, c='w',
            horizontalalignment='center', verticalalignment='center')
    start = plt.Rectangle((x0-1, y0-1), 2, 2, color='w')
    ax.add_patch(start)

    # Text annotations
    ax.text(0, 0, r'$\mathcal{N}$', fontsize=10,
            horizontalalignment='center', verticalalignment='center')
    ax.text(40, 0, r'$\mathcal{G}$', fontsize=10,
            horizontalalignment='center', verticalalignment='center')

    plt.legend()

    plt.subplots_adjust(
        top=0.94,
        bottom=0.095,
        left=0.08,
        right=0.975,
        hspace=0.4,
        wspace=0.2
    )

    plt.show()
    # plt.savefig('trajectory_init({}, {}).pdf'.format(x0, y0))


def create_environment(params):
    width, height = params['env_dimension']
    obs_pos = params['obstacle_coord']
    obs_diameter = params['obstacle_diameter']
    obj_pos = params['objective_coord']
    obj_radius = params['objective_diameter']
    occlusion_rate = params['occlusion_rate']

    environment = SimpleEnvironment(width, height, occlusion_rate)
    environment.add_agent(obs_pos, obs_diameter, shape='circle')
    environment.add_agent(obj_pos, obj_radius, shape='circle', color=(1.0, 0.0, 0.0))

    return environment


def load_model(params):
    model_path = params['model_path']

    model = keras.models.load_model(model_path)

    return model


if __name__ == '__main__':

    # MODEL
    # x(t + 1) = x(t) + s * ux
    # y(t + 1) = y(t) + s * uy
    # ux = -a(x - xt) + b(x - 1);
    # uy = -a(y - yt) + b(y - 2);

    def img_preprocessing(img):

        img = img.astype(np.uint8).astype('float32')
        img = np.expand_dims(img, axis=0)

        return img


    sim_params = dict(
        env_dimension=(100, 60),
        obstacle_coord=(0, 0),
        obstacle_diameter=16,
        objective_coord=(40, 0),
        objective_diameter=8,
        stop_difference=1.0,
        occlusion_rate=0.0,
        preprocessing=img_preprocessing,
        model_path='/home/alejandro/Documents/Projects/Navigation/Linking_Perception_to_Control/models/saved_models/cnn_100kTrain',
        use_learning=False
    )

    for i in range(1):
        main(sim_params)
