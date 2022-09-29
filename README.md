# Data-Assisted Vision-Based Hybrid Control for Robust Stabilization with Obstacle Avoidance via Learning of Perception Maps

Code for the paper `Data-Assisted Vision-Based Hybrid Control for Robust Stabilization with Obstacle Avoidance via Learning of Perception Maps` [ACC 2022](https://ieeexplore.ieee.org/abstract/document/9867532)

We study the problem of target stabilization with robust obstacle avoidance in robots and vehicles that have access only to vision-based sensors for the purpose of real-time localization. This problem is particularly challenging due to the topological obstructions induced by the obstacle, which preclude the existence of smooth feedback controllers able to achieve simultaneous stabilization and robust obstacle avoidance. To overcome this issue, we develop a vision-based hybrid controller that switches between two different feedback laws depending on the current position of the vehicle using a hysteresis mechanism and a data-assisted supervisor. The main innovation of the paper is the incorporation of suitable perception maps into the hybrid controller. These maps can be learned from data obtained from cameras in the vehicles and trained via convolutional neural networks (CNN). Under suitable assumptions on this perception map, we establish theoretical guarantees for the trajectories of the vehicle in terms of convergence and obstacle avoidance. Moreover, the proposed vision-based hybrid controller is numerically tested under different scenarios, including noisy data, sensors with failures, and cameras with occlusions.
