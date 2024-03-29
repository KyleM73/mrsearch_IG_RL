import numpy as np
import torch

def dict2torch(x, device):
        # assumes two levels
        obs = []
        for k,v in x.items():
            if isinstance(v,dict):
                obs.append(dict2torch(v, device))
            else:
                if isinstance(v,np.ndarray):
                    obs.append(torch.from_numpy(v).to(device))
                elif isinstance(v,torch.Tensor):
                    obs.append(v.to(device))
                else:
                    obs.append(v)
        return tuple(obs)

def torch2obsdict(x, env):
    obs = {}
    for i,agent in zip(range(len(x)),env.possible_agents):
        obs[agent] = {
            "img" : x[i][0],#.numpy().astype(np.float32),
            "vec" : x[i][1].numpy().astype(np.float32)
            }
    return obs

def torch2dict(x, env):
    obs = {}
    for i,agent in zip(range(len(x)),env.possible_agents):
        obs[agent] = x[i]
    return obs

def bresenham(start,end):
    """
    Adapted from PythonRobotics:
    https://github.com/AtsushiSakai/PythonRobotics/blob/master/Mapping/lidar_to_grid_map/lidar_to_grid_map.py

    Implementation of Bresenham's line drawing algorithm
    See en.wikipedia.org/wiki/Bresenham's_line_algorithm
    Bresenham's Line Algorithm
    Produces a list from start and end (original from roguebasin.com)
    >>> points1 = bresenham((4, 4), (6, 10))
    >>> print(points1)
    [[4,4], [4,5], [5,6], [5,7], [5,8], [6,9], [6,10]]
    """
    # setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    is_steep = abs(dy) > abs(dx)  # determine how steep the line is
    if is_steep:  # rotate line
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    # swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    dx = x2 - x1  # recalculate differentials
    dy = y2 - y1  # recalculate differentials
    error = int(dx / 2.0)  # calculate error
    y_step = 1 if y1 < y2 else -1
    # iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = [y, x] if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += y_step
            error += dx
    if swapped:  # reverse the list if the coordinates were swapped
        points.reverse()
    return points[1:-1] #do not include endpoints