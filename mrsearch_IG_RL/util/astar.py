from mrsearch_IG_RL.external import *

class Node:
    def __init__(self):
        self.position = (0, 0)
        self.parent = None

        self.g = 0
        self.h = 0
        self.f = 0

    def set_position(self,pose):
        self.position = pose

    def get_position(self):
        return self.position

    def set_parent(self,parent):
        self.parent = parent

    def get_parent(self):
        return self.parent

    def set_g(self,g):
        self.g = g

    def get_g(self):
        return self.g

    def set_h(self,h):
        self.h = h

    def get_h(self):
        return self.h

    def set_f(self,f):
        self.f = f

    def get_f(self):
        return self.f

    def same_pose(self,node):
        return self.position == node.get_position()

    def __hash__(self):
        return hash((self.position,self.parent))

    def __eq__(self,node):
        if self.position == node.get_position() and self.parent == node.get_parent():
            return True
        return False

    def __repr__(self):
        #return "Position: {pose}".format(pose=self.position)
        return "(Position: {pose} , Parent: {parent})".format(pose=self.position, parent=self.parent)

class Grid:
    def __init__(self,grid):
        assert isinstance(grid,np.ndarray)
        self.grid = grid
        self.rlim,self.clim = self.grid.shape
        self.neighbors =  [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
                ]

    def get_value(self,i,j):
        return self.grid[i,j]

    def get_shape(self):
        return self.grid.shape

    def get_adjacent(self,node):
        r_,c_ = node.get_position()
        adjacent = []
        for nn in self.neighbors:
            flag = True
            r = r_ + nn[0]
            c = c_ + nn[1]
            if 0 <= r < self.rlim and 0 <= c < self.clim and not self.get_value(r,c):
                n = Node()
                n.set_position((r,c))
                n.set_parent(node)
                adjacent.append(n)
        return adjacent

    def is_open(self,pose):
        return not bool(self.get_value(*pose))

    def __repr__(self):
        return "{}".format(self.grid)

def A_Star(grid,start_rc,target_rc):
    if not isinstance(start_rc,tuple):
        start_rc = tuple(start_rc)
    if not isinstance(target_rc,tuple):
        target_rc = tuple(target_rc)
    start = Node()
    start.set_position(start_rc)
    target = Node()
    target.set_position(target_rc)

    if not grid.is_open(start.get_position()) or not grid.get_adjacent(start):
        print("Invalid Start Position:")
        print(start)
        return #[]
    elif not grid.is_open(target.get_position()) or not grid.get_adjacent(target):
        print("Invalid Target:")
        print(target)
        return #[start.get_position()]

    _open = []
    _closed = []
    _open.append(start)

    max_iters = grid.get_shape()[0] * grid.get_shape()[1] 
    cnt = 0

    while _open and cnt < max_iters:
        cnt += 1

        min_f = np.argmin([n.get_f() for n in _open])
        current = _open.pop(min_f)
        if current.get_position() in _closed:
            continue
        _closed.append(current.get_position())
        if current.same_pose(target):
            #print("Solution found.")
            break
        neighbors = grid.get_adjacent(current)
        for n in neighbors:
            if n.get_position() in _closed:
                continue
            n.set_g(current.get_g() + 1)
            x1, y1 = n.get_position()
            x2, y2 = target.get_position()
            n.set_h((y2 - y1) ** 2 + (x2 - x1) ** 2)
            n.set_f(n.get_h() + n.get_g())
            _open.append(n)
    else:
        #print("Solution not found.")
        return #[start.get_position()]
    path = []
    while current.get_parent() is not None:
        path.append(current.get_position())
        current = current.get_parent()
    path.append(current.get_position())
    return path[::-1]