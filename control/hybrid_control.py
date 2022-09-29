from copy import deepcopy
import os

from utils.generate_data import build_gif
from control import load_model, create_environment, controller
from control2 import plot_trajectory_normal_controller
from models.linear_model import LinearModel

from tensorflow import make_tensor_proto, make_ndarray

import matplotlib.pyplot as plt
import numpy as np
import moviepy.video.io.ImageSequenceClip


def main(params):

    trajectory, init_x, init_y, diff = waypoint_tracking(params)

    # build_gif(trajectory, 'pred_init({:.1f}, {:.1f})_diff_{:.2f}.gif'.format(init_x, init_y, diff))


# def cost_function2(current_pos, obj_pos, fringe, obs_radius):
#
#     obs_pos = (0, 0)
#     obs_diameter = 16
#
#     obs_radius = 0.5 * obs_diameter
#     x = -((current_pos[0] - obj_pos[0]) ** 2)
#     y = -((current_pos[1] - obj_pos[1]) ** 2)
#
#     if (current_pos[0] - obs_pos[0])**2 + (current_pos[1] - obs_pos[1])**2 > obs_radius**2:
#         dist = np.sqrt((current_pos[0] - obs_pos[0]) ** 2 + (current_pos[1] - obs_pos[1]) ** 2) - obs_radius
#     else:
#         dist = 0
#
#     if dist <= obs_radius:
#         b = (np.log(obs_radius) - np.log(dist)) * ((dist - obs_radius) ** 2)
#     else:
#         b = 0
#
#     return x + y - b


def cost_function(state, target, fringe, obs_radius):

    x = 0.5 * ((state[0] - target[0])**2)
    y = 0.5 * ((state[1] - target[1])**2)

    dist = fringe.distance(state)   # - np.sqrt(obs_radius)
    # obs_radius = np.sqrt(obs_radius)
    if dist == np.inf:
        b = dist
    elif 0 <= dist <= obs_radius:
        # b = (np.log(1) - np.log(dist)) * ((dist - obs_radius)**2)
        b = (np.log(obs_radius) - np.log(dist)) * ((dist - obs_radius)**2)
    else:
        b = 0

    return x + y + b


def gradient_controller(params):

    state = params['state']
    current_pos = state[-1]
    obj_pos = params['obj_pos']
    if isinstance(obj_pos, list):
        obj_pos = obj_pos[-1]

    # obs_pos = params['obstacle_coord']
    obs_radius = params['obstacle_diameter'] / 2
    current_controller = params['current_controller']
    chi, lamb = params['chi'], params['lamb']

    # Fringes
    fringe_O1 = params['fringe_O1']
    fringe_O2 = params['fringe_O2']
    v1 = cost_function(current_pos, obj_pos, fringe_O1, obs_radius)
    v2 = cost_function(current_pos, obj_pos, fringe_O2, obs_radius)

    if current_controller == 1:
        if v1 >= (chi - lamb) * v2:  # Switch controllers
            params['current_controller'] = 2
            v = v2
            fringe = fringe_O2
        else:
            v = v1
            fringe = fringe_O1
    else:
        if v2 >= (chi - lamb) * v1:  # Switch controllers
            params['current_controller'] = 1
            v = v1
            fringe = fringe_O1
        else:
            v = v2
            fringe = fringe_O2
    params['controller_history'].append(params['current_controller'])

    def get_jacobian():

        h = 0.01
        cost = v

        jacobian = []
        for i in range(len(current_pos)):

            mod_params = current_pos.copy()
            mod_params[i] = mod_params[i] + h

            # cost_h = cost_function(mod_params, obj_pos, obs_pos, obs_diameter/2)
            cost_h = cost_function(mod_params, obj_pos, fringe, obs_radius)

            partial_derivative = (cost_h - cost) / h

            jacobian.append(partial_derivative)

        jacobian = np.array(jacobian)
        jacobian = np.clip(jacobian, -10, 10)  # Gradient clipping

        return jacobian

    dt = params['dt']
    u = -get_jacobian()

    saddle_point = [-13.5, 0.5]
    delta_x = 0.5
    delta_y = 0.5
    delta_x = delta_x if abs(current_pos[0] - saddle_point[0]) <= delta_x else 0.0
    delta_y = delta_y if abs(current_pos[1] - saddle_point[1]) <= delta_y else 0.0

    print(delta_x, delta_y)

    eps = 1e-2

    x = current_pos[0] + dt * (u[0] - eps * delta_x * (current_pos[0] - saddle_point[0]))
    y = current_pos[1] + dt * (u[1] - eps * delta_y * (current_pos[1] - saddle_point[1]))

    # x = current_pos[0] + dt * u[0]
    # y = current_pos[1] + dt * u[1]

    # if params['id'] == 'reference':
    #     limit = 0.5
    #     x += np.random.normal(loc=0.0, scale=limit)
    #     y += np.random.normal(loc=0.0, scale=limit)
    #     x += np.random.uniform(low=-limit, high=limit)
    #     y += np.random.uniform(low=-limit, high=limit)

    params['state'].append([x, y])

    return params


def waypoint_tracking(params):

    # Parameter retrieval
    obs_pos = params['obstacle_coord']
    obs_diameter = params['obstacle_diameter']
    objective_pos = params['objective_coord']
    stop_difference = params['stop_difference']
    preprocessing = params['preprocessing']

    # Convenience Functions
    def calculate_difference(params):

        pos = params['state'][-1]
        obj = params['obj_pos']
        if isinstance(obj, list):
            obj = obj[-1]

        return np.sqrt((pos[0] - obj[0]) ** 2 + (pos[1] - obj[1]) ** 2)

    def get_prediction(pred):
        return list(make_ndarray(make_tensor_proto(pred))[0])

    # Set agent's initial position (coordinates)
    init_x = -40  # np.random.randint(low=-width//2, high=0)  #width//2)
    init_y = 0   # np.random.randint(low=-height//2, high=height//2)

    # Initialize the state at the agent's initial position
    environment = create_environment(params)
    model = load_model(params)

    # Controller parameters
    reference_params = dict(
        a=0.9, dt=0.05,
        chi=1.1, lamb=0.09,
        current_controller=1,
        controller_history=[1],
        state=[[init_x, init_y]],
        obj_pos=[objective_pos],
        obstacle_coord=obs_pos,
        obstacle_diameter=obs_diameter,
        color=(1.0, 1.0, 1.0),
        id='reference'
    )
    reference_params['b'] = 1 - reference_params['a']

    # Define Fringe Sets
    obs_pos = reference_params['obstacle_coord']
    diameter = reference_params['obstacle_diameter']
    obs_radius = (diameter // 2) + (diameter // 4)
    p0 = (obs_pos[0] - obs_radius, obs_pos[1])
    p2 = (obs_pos[0] + obs_radius, obs_pos[1])
    reference_params['fringe_O1'] = Fringe(  # -
        p0=p0, p1=(obs_pos[0], obs_pos[1] - obs_radius), p2=p2
    )
    reference_params['fringe_O2'] = Fringe(  # +
        p0=p0, p1=(obs_pos[0], obs_pos[1] + obs_radius), p2=p2
    )

    tracker_params = deepcopy(reference_params)
    # tracker_params['dt'] = 0.2
    tracker_params['color'] = (0.0, 1.0, 0.0)
    tracker_params['obj_pos'] = [reference_params['state'][0]]
    tracker_params['id'] = 'tracker'

    i = 0
    trajectory = []
    while calculate_difference(reference_params) > stop_difference:
        i += 1
        if i > 300:
            break

        d = tracker_params['state'][-1]    # reference_params['state'][-1]
        drawing = environment.draw_trajectory([d], color=tracker_params['color'], ignore_occlusion=True)[0]
        d2 = reference_params['state'][-1]
        drawing2 = environment.draw_trajectory([d2])[0]

        d_comb = np.clip(drawing + drawing2, 0.0, 1.0)
        trajectory.append(d_comb)

        # Get Previous state
        x_t, y_t = reference_params['state'][-1]

        ##################
        data_transmission_stop = 60   # Timestep when the follower stops getting new info. about the reference's state.
        failure_rate = 0.0    # failure_rate = 0 means that the camera never fails.
        # if i < data_transmission_stop and np.random.uniform() >= failure_rate:
        if np.random.uniform() >= failure_rate:
            # When using prediction
            drawing = environment.draw_trajectory([[x_t, y_t]])  # Take picture of environment to predict objective's state.
            prev_state = preprocessing(drawing[0])
            predicted_state = model(prev_state)
            predicted_state = get_prediction(predicted_state)

            # When using oracle:
            # predicted_state = [x_t, y_t]
        else:
            if isinstance(tracker_params['obj_pos'], list):
                predicted_state = tracker_params['obj_pos'][-1]
            else:
                predicted_state = tracker_params['obj_pos']

        if isinstance(tracker_params['obj_pos'], list):
            tracker_params['obj_pos'].append(predicted_state)
        else:
            tracker_params['obj_pos'] = predicted_state

        print('Iteration {}:\n\tReference = ({}, {})\tPrediction = ({}, {})'.format(i, x_t, y_t, predicted_state[0], predicted_state[1]))
        ##################

        # Move Tracker
        # tracker_params = gradient_controller(tracker_params)

        # Move Reference
        reference_params = gradient_controller(reference_params)

    print('Waypoint Tracking stopped in iteration {}'.format(i))
    diff = calculate_difference(reference_params)

    # plot_trajectory_normal_controller(reference_params, tracker_params)
    # plot_trajectory_hc(reference_params, tracker_params)
    # video_contour_plot(reference_params, tracker_params)
    plot_prediction_vs_label(reference_params, tracker_params)

    return trajectory, init_x, init_y, diff


def plot_prediction_vs_label(reference_params, tracker_params):

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
    plt.suptitle('State Prediction vs. Real State')
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
    plt.scatter(x, y, edgecolors='w', facecolors='none', label='Real/Target State')

    y_upper = [yi + 3.8 for _, yi in reference_params['state']]
    y_lower = [yi - 3.8 for _, yi in reference_params['state']]
    plt.plot(x, y_upper, c='y', ls=':')
    plt.plot(x, y_lower, c='y', ls=':')

    x = [xi for xi, _ in tracker_params['obj_pos']]
    y = [yi for _, yi in tracker_params['obj_pos']]
    plt.scatter(x, y, label='Predicted State', color='g', marker='x')

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


def plot_trajectory_hc(reference_params, tracker_params):

    plt.subplots(2, 1, figsize=(7, 7), gridspec_kw={'height_ratios': [2.5, 1]})
    ax1 = plt.subplot(2, 1, 1)
    ax1.grid(True, ls=':', lw=0.5, alpha=0.5)
    ax1.set_axisbelow(True)

    # Static elements
    obstacle = plt.Circle((0, 0), radius=8, ls='-', color='w')
    objective = plt.Circle((40, 0), radius=4, color='r')
    ax1.add_patch(obstacle)
    ax1.add_patch(objective)

    # Labels
    plt.title('Agents\' trajectory')
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')

    # Plot config.
    ax1.set_facecolor('k')
    ax1.set_xlim(left=-50, right=50)
    ax1.set_ylim(bottom=-30, top=30)
    plt.xticks(ticks=np.arange(-50, 50, 10))
    plt.yticks(ticks=np.arange(-30, 30, 10))

    x = [xi for xi, _ in reference_params['state']]
    y = [yi for _, yi in reference_params['state']]

    # for i in range(10, len(y)//3):
    #     y[i] += i / 3
    # for i in range(len(y) // 3, len(y)):
    #     y[i] += 25    # 10 when (-12, 2) ;; 25 when (-37, -17)

    plt.plot(x, y, c='w', label='Leader')

    x = [xi for xi, _ in tracker_params['state']]
    y = [yi for _, yi in tracker_params['state']]
    plt.plot(x, y, c='g', label='Follower', ls='--')

    x0, y0 = reference_params['state'][0]
    ax1.text(x0+1, y0-3, r'$({}, {})$'.format(x0, y0), fontsize=8, c='w',
            horizontalalignment='center', verticalalignment='center')
    start = plt.Rectangle((x0-1, y0-1), 2, 2, color='w')
    ax1.add_patch(start)

    # Text annotations
    ax1.text(0, 0, r'$\mathcal{N}$', fontsize=10,
            horizontalalignment='center', verticalalignment='center')
    ax1.text(40, 0, r'$\mathcal{G}$', fontsize=10,
            horizontalalignment='center', verticalalignment='center')

    plt.legend()

    ###########################
    ax2 = plt.subplot(2, 1, 2)
    ax2.set_facecolor('k')
    ax2.grid(True, ls=':', lw=0.5, alpha=0.5)

    ax2.plot(reference_params['controller_history'], c='w')
    ax2.plot(tracker_params['controller_history'], c='g', ls='--')

    plt.title('Agents\' controller state')
    plt.xlabel('Timestep')
    plt.ylabel(r'Logic State $q_i$')
    plt.yticks([1, 2])

    plt.subplots_adjust(
        top=0.94,
        bottom=0.095,
        left=0.125,
        right=0.955,
        hspace=0.4,
        wspace=0.2
    )

    plt.show()
    # plt.savefig('trajectory_init({}, {}).pdf'.format(x0, y0))


def draw_countor_plot():

    draw_countor_plot_q1()
    draw_countor_plot_q2()


def video_contour_plot(reference_params, tracker_params):

    controllers_indices = [1, 2]
    frame_rates = [5, 24]

    ref_state = reference_params['state']
    tracker_state = tracker_params['state']
    tracker_obj = tracker_params['obj_pos']

    folder_name_base = 'frames_FollowerLevelSets_q{}_{}'
    image_folders = [folder_name_base.format(q, ref_state[0]) for q in controllers_indices]
    for folder in image_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    state_range = min(len(ref_state), len(tracker_state))
    for i in range(state_range):
        reference_agent = (ref_state[i], 'w', 2, 'Leader')
        follower_agent = (tracker_state[i], 'g', 2, 'Follower')
        agents = [reference_agent, follower_agent]

        for q in controllers_indices:
            folder = image_folders[q-1]

            cost_point = tracker_obj[i] if isinstance(tracker_obj, list) else tracker_obj
            if q == 1:
                draw_countor_plot_q1(cost_point, agents, save_path='{}/{}.png'.format(folder, i))
            elif q == 2:
                draw_countor_plot_q2(cost_point, agents, save_path='{}/{}.png'.format(folder, i))

    video_name_base = 'FollowerLevelSets_q{}_{}_{}fps.mp4'
    for q in controllers_indices:
        folder = image_folders[q-1]
        image_files = [folder + '/' + img for img in os.listdir(folder) if img.endswith(".png")]
        image_files = sorted(image_files, key=lambda name: int(name.split('/')[-1].split('.')[0]))

        for fps in frame_rates:
            video_name = video_name_base.format(q, ref_state[0], fps)
            print('Saving:', video_name)
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
            clip.write_videofile(video_name)


def draw_countor_plot_q1(cost_point=(40, 0), agents=None, save_path=None):

    if save_path is None:
        fig = plt.figure(figsize=(60, 100))
    else:
        fig = plt.figure(figsize=(25, 13))
    ax = plt.gca()
    ax.grid(True, ls=':', lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    obs_pos = (0, 0)
    diameter = 16
    obs_radius = (diameter // 2) + (diameter // 4)
    p0 = (obs_pos[0] - obs_radius, obs_pos[1])
    p2 = (obs_pos[0] + obs_radius, obs_pos[1])
    fringe1 = Fringe(
        p0=p0, p1=(obs_pos[0], obs_pos[1] - obs_radius), p2=p2
    )
    fringe2 = Fringe(
        p0=p0, p1=(obs_pos[0], obs_pos[1] + obs_radius), p2=p2
    )

    sampling = 5
    x = np.linspace(-50, 50, sampling*100)
    y = np.linspace(30, -30, sampling*60)

    mesh_shape = (len(y), len(x))
    mesh1 = np.zeros(mesh_shape)
    mesh2 = np.zeros(mesh_shape)

    fringe1_borders = []
    flow_set = []  # C
    jump_set = []  # D

    chi, lamb = 1.1, 0.09
    for i, x_i in enumerate(x):
        c, d = None, None
        for j, y_j in enumerate(y):
            # mesh1[j, i] = cost_function((x_i, y_j), (40, 0), fringe1, 8)
            # mesh2[j, i] = cost_function((x_i, y_j), (40, 0), fringe2, 8)
            mesh1[j, i] = cost_function((x_i, y_j), cost_point, fringe1, 8)
            mesh2[j, i] = cost_function((x_i, y_j), cost_point, fringe2, 8)

            ############# FRINGE 1 IS VALLEY ######################################################################
            # Fill Flow Set (C)
            if c is None and mesh1[j, i] != np.inf and mesh1[j, i] <= chi * mesh2[j, i]:
                c = y_j

            # Fill Jump Set (D)
            if d is None and mesh1[j, i] < (chi - lamb) * mesh2[j, i]:
                d = y_j
            ########################################################################################################

        fringe1_borders.append(fringe1(x_i))
        flow_set.append(c)
        jump_set.append(d)

    plt.contour(x, y, mesh1, cmap='Wistia', levels=16)
    plt.plot(x, fringe1_borders, ls='--', color='silver')

    flow_set_color = 'fuchsia'
    jump_set_color = 'cyan'
    plt.plot(x, flow_set, c=flow_set_color, lw=2)  # 0.5)
    plt.plot(x, jump_set, c=jump_set_color, lw=2)  # 0.5)

    flow_set_arrows = [70, 110, 150]
    for arrow in flow_set_arrows:
        plt.arrow(x[arrow], flow_set[arrow], -1, -1, color=flow_set_color, head_width=0.5)
        plt.arrow(x[-arrow], flow_set[-arrow], 1, -1, color=flow_set_color, head_width=0.5)

    arrow = 60  # 150
    ax.text(x[arrow] + 0.0, jump_set[arrow] - 2, r'$\mathit{C}_1$', fontsize=24, c=flow_set_color,
                horizontalalignment='center', verticalalignment='center')
    ax.text(x[-arrow] - 0.0, jump_set[-arrow] - 2, r'$\mathit{C}_1$', fontsize=24, c=flow_set_color,
            horizontalalignment='center', verticalalignment='center')

    jump_set_arrows = [50, 90, 130]
    for arrow in jump_set_arrows:
        plt.arrow(x[arrow], jump_set[arrow], 1, 1, color=jump_set_color, head_width=0.5)
        plt.arrow(x[-arrow], jump_set[-arrow], -1, 1, color=jump_set_color, head_width=0.5)

    arrow = 90
    ax.text(x[arrow] + 0.2, jump_set[arrow] + 2.5, r'$\mathit{D}_1$', fontsize=24, c=jump_set_color,
                horizontalalignment='center', verticalalignment='center')
    ax.text(x[-arrow] - 0.2, jump_set[-arrow] + 2.5, r'$\mathit{D}_1$', fontsize=24, c=jump_set_color,
            horizontalalignment='center', verticalalignment='center')

    # plt.contour(x, y, mesh2)
    # plt.matshow(mesh1, cmap='jet')

    # Static elements
    obstacle = plt.Circle((0, 0), radius=8, ls='-', color='w')
    objective = plt.Circle((40, 0), radius=4, color='r')
    ax.add_patch(obstacle)
    ax.add_patch(objective)

    if agents is not None and isinstance(agents, list):
        added_labels = set()
        for coord, color, radius, label in agents:
            x = coord[0] - (radius / 2)
            y = coord[1] - (radius / 2)
            if label not in added_labels:
                ag = plt.Rectangle((x,y), radius, radius, color=color, label=label)
            else:
                ag = plt.Rectangle((x, y), radius, radius, color=color)
            added_labels.add(label)
            ax.add_patch(ag)
        plt.legend(fontsize=16)

    # Text annotations
    ax.text(0, 0, r'$\mathcal{N}$', fontsize=26,
            horizontalalignment='center', verticalalignment='center')
    ax.text(40, 0, r'$\mathcal{G}$', fontsize=26,
            horizontalalignment='center', verticalalignment='center')
    ax.text(0, 25, r'$V_1 (p) = \mathcal{\infty}$', fontsize=26, color='w',
            horizontalalignment='center', verticalalignment='center')
    ax.text(0, -20, r'$\mathcal{O}_1$', fontsize=26, color='w',
            horizontalalignment='center', verticalalignment='center')

    # Labels
    plt.title(r'Level sets of the localization function when $q = 1$', fontsize=20)
    plt.xlabel('x-coordinate', fontsize=16)
    plt.ylabel('y-coordinate', fontsize=16)

    # Plot config.
    ax.set_facecolor('k')
    ax.set_xlim(left=-50, right=50)
    ax.set_ylim(bottom=-30, top=30)
    plt.xticks(ticks=np.arange(-50, 50, 10))
    plt.yticks(ticks=np.arange(-30, 30, 10))
    fig.subplots_adjust(
        top=0.95,
        bottom=0.05,
        left=0.035,
        right=1.0,
        hspace=0.2,
        wspace=0.2
    )
    cb = plt.colorbar()
    cb.lines[0].set_linewidth(10)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, orientation='landscape')


def draw_countor_plot_q2(cost_point=(40, 0), agents=None, save_path=None):

    if save_path is None:
        fig = plt.figure(figsize=(60, 100))
    else:
        fig = plt.figure(figsize=(25, 13))
    ax = plt.gca()
    ax.grid(True, ls=':', lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    obs_pos = (0, 0)
    diameter = 16
    obs_radius = (diameter // 2) + (diameter // 4)
    p0 = (obs_pos[0] - obs_radius, obs_pos[1])
    p2 = (obs_pos[0] + obs_radius, obs_pos[1])
    fringe1 = Fringe(
        p0=p0, p1=(obs_pos[0], obs_pos[1] + obs_radius), p2=p2
    )
    fringe2 = Fringe(
        p0=p0, p1=(obs_pos[0], obs_pos[1] - obs_radius), p2=p2
    )

    sampling = 5
    x = np.linspace(-50, 50, sampling*100)
    y = np.linspace(30, -30, sampling*60)

    mesh_shape = (len(y), len(x))
    mesh1 = np.zeros(mesh_shape)
    mesh2 = np.zeros(mesh_shape)

    fringe1_borders = []
    flow_set = []  # C
    jump_set = []  # D

    chi, lamb = 1.1, 0.09
    for i, x_i in enumerate(x):
        c, d = None, None
        for j, y_j in enumerate(y):
            # mesh1[j, i] = cost_function((x_i, y_j), (40, 0), fringe1, 8)
            # mesh2[j, i] = cost_function((x_i, y_j), (40, 0), fringe2, 8)
            mesh1[j, i] = cost_function((x_i, y_j), cost_point, fringe1, 8)
            mesh2[j, i] = cost_function((x_i, y_j), cost_point, fringe2, 8)

            ############# FRINGE 1 IS MOUNTAIN #####################################################################
            # Fill Flow Set (C)
            if mesh1[j, i] != np.inf and mesh1[j, i] <= chi * mesh2[j, i]:
                c = y_j

            # Fill Jump Set (D)
            if mesh1[j, i] < (chi - lamb) * mesh2[j, i]:
                d = y_j
            ########################################################################################################

        fringe1_borders.append(fringe1(x_i))
        flow_set.append(c)
        jump_set.append(d)

    plt.contour(x, y, mesh1, cmap='Wistia', levels=16)
    plt.plot(x, fringe1_borders, ls='--', color='silver')

    flow_set_color = 'fuchsia'
    jump_set_color = 'cyan'
    plt.plot(x, flow_set, c=flow_set_color, lw=2)  # 0.5)
    plt.plot(x, jump_set, c=jump_set_color, lw=2)  # 0.5)

    flow_set_arrows = [70, 110, 150]
    for arrow in flow_set_arrows:
        plt.arrow(x[arrow], flow_set[arrow], -1, 1, color=flow_set_color, head_width=0.5)
        plt.arrow(x[-arrow], flow_set[-arrow], 1, 1, color=flow_set_color, head_width=0.5)

    arrow = 60  # 150
    ax.text(x[arrow] - 0.75, jump_set[arrow] + 2, r'$\mathit{C}_2$', fontsize=24, c=flow_set_color,
                horizontalalignment='center', verticalalignment='center')
    ax.text(x[-arrow] + 0.75, jump_set[-arrow] + 2, r'$\mathit{C}_2$', fontsize=24, c=flow_set_color,
            horizontalalignment='center', verticalalignment='center')

    jump_set_arrows = [50, 90, 130]
    for arrow in jump_set_arrows:
        plt.arrow(x[arrow], jump_set[arrow], 1, -1, color=jump_set_color, head_width=0.5)
        plt.arrow(x[-arrow], jump_set[-arrow], -1, -1, color=jump_set_color, head_width=0.5)

    arrow = 90
    ax.text(x[arrow] + 0.5, jump_set[arrow] - 2.5, r'$\mathit{D}_2$', fontsize=24, c=jump_set_color,
                horizontalalignment='center', verticalalignment='center')
    ax.text(x[-arrow] - 0.5, jump_set[-arrow] - 2.5, r'$\mathit{D}_2$', fontsize=24, c=jump_set_color,
            horizontalalignment='center', verticalalignment='center')

    # flow_set = [f for f in flow_set if f is not None]
    # jump_set = [j for j in jump_set if j is not None]
    # print(len(x), len(flow_set), len(jump_set))

    # plt.contour(x, y, mesh2)
    # plt.matshow(mesh1, cmap='jet')

    # Static elements
    obstacle = plt.Circle((0, 0), radius=8, ls='-', color='w')
    objective = plt.Circle((40, 0), radius=4, color='r')
    ax.add_patch(obstacle)
    ax.add_patch(objective)

    if agents is not None and isinstance(agents, list):
        added_labels = set()
        for coord, color, radius, label in agents:
            x = coord[0] - (radius / 2)
            y = coord[1] - (radius / 2)
            if label not in added_labels:
                ag = plt.Rectangle((x,y), radius, radius, color=color, label=label)
            else:
                ag = plt.Rectangle((x, y), radius, radius, color=color)
            added_labels.add(label)
            ax.add_patch(ag)
        plt.legend(fontsize=16)

    # Text annotations
    ax.text(0, 0, r'$\mathcal{N}$', fontsize=26,
            horizontalalignment='center', verticalalignment='center')
    ax.text(40, 0, r'$\mathcal{G}$', fontsize=26,
            horizontalalignment='center', verticalalignment='center')
    ax.text(0, -20, r'$V_2 (p) = \mathcal{\infty}$', fontsize=26, color='w',
            horizontalalignment='center', verticalalignment='center')
    ax.text(0, 25, r'$\mathcal{O}_2$', fontsize=26, color='w',
            horizontalalignment='center', verticalalignment='center')

    # Labels
    plt.title(r'Level sets of the localization function when $q = 2$', fontsize=20)
    plt.xlabel('x-coordinate', fontsize=16)
    plt.ylabel('y-coordinate', fontsize=16)

    # Plot config.
    ax.set_facecolor('k')
    ax.set_xlim(left=-50, right=50)
    ax.set_ylim(bottom=-30, top=30)
    plt.xticks(ticks=np.arange(-50, 50, 10))
    plt.yticks(ticks=np.arange(-30, 30, 10))
    fig.subplots_adjust(
        top=0.95,
        bottom=0.05,
        left=0.035,
        right=1.0,
        hspace=0.2,
        wspace=0.2
    )
    cb = plt.colorbar()
    cb.lines[0].set_linewidth(10)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, orientation='landscape')


class Fringe:

    def __init__(self, p0, p1, p2):

        assert (p0[1] < p1[1] and p1[1] > p2[1]) or (p0[1] > p1[1] and p1[1] < p2[1])

        self.p0 = p0
        self.p1 = p1
        self.p2 = p2

        self.mountain_fringe = self.p0[1] < self.p1[1]
        self.f1 = LinearModel(p0, p1)
        self.f2 = LinearModel(p1, p2)

    def __call__(self, x):

        return self.f1(x) if x <= self.p1[0] else self.f2(x)

    def distance(self, p):

        x1, y1 = self.f1.intersection_point(p)
        x2, y2 = self.f2.intersection_point(p)

        def below_function(func, point):
            return ((point[1] - func.bias) / func.slope) < point[0]

        if self.mountain_fringe:
            if y1 >= self.p1[1] and y2 >= self.p1[1]:
                return self._euclidean_distance(p, self.p1)
            elif (not below_function(self.f1, p)) and p[0] <= self.p1[0]:
                return self._euclidean_distance(p, (x1, y1))
            elif below_function(self.f2, p) and p[0] >= self.p1[0]:
                return self._euclidean_distance(p, (x2, y2))
            else:
                return np.inf
        else:
            if y1 <= self.p1[1] and y2 <= self.p1[1]:
                return self._euclidean_distance(p, self.p1)
            elif (not below_function(self.f1, p)) and p[0] <= self.p1[0]:
                return self._euclidean_distance(p, (x1, y1))
            elif below_function(self.f2, p) and p[0] >= self.p1[0]:
                return self._euclidean_distance(p, (x2, y2))
            else:
                return np.inf

    @staticmethod
    def _euclidean_distance(p0, p1):
        x0, y0 = p0
        x1, y1 = p1
        dist = np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        return dist


if __name__ == '__main__':

    def img_preprocessing(img):

        img = img.astype(np.uint8).astype('float32')
        img = np.expand_dims(img, axis=0)

        return img

    def get_model_path(model_name):
        base = '/home/alejandro/Documents/Projects/Navigation/Linking_Perception_to_Control/models/saved_models/'
        return base + model_name

    sim_params = dict(
        env_dimension=(100, 60),
        obstacle_coord=(0, 0),
        obstacle_diameter=16,
        objective_coord=(40, 0),
        objective_diameter=8,
        stop_difference=0.5,
        occlusion_rate=0.0,  # 1.0,
        preprocessing=img_preprocessing,
        model_path=get_model_path('cnn_100kTrain'),    # _100%_OcclusionRate'),
        use_learning=False
    )

    for i in range(1):
        main(sim_params)

    # draw_countor_plot()
