import numpy as np 
import matplotlib.pyplot as plt

def count(map):
    '''
        use loop instead of recursion for easy debugging and code is more comprehensive
    '''
    assert map.max() == 1 and map.min() == 0, "Map must be binary."
    assert len(map.shape) == 2 or (len(map.shape) == 3 and  map.shape[2] == 1), "Requires 2D array."

    if len(map.shape) == 3 : 
        map = np.squeeze(map, axis=2)

    island_map = -1 * np.ones(shape=map.shape, dtype=np.int8)

    await_pts = []
    for y in range(map.shape[0]):
        for x in range(map.shape[1]):
            if not map[y,x] == 0:
                await_pts.append([y,x])

    isl_idx = 0 # island idx starts from 1
    isl_pts = []
    bbs = [] # bounding-box list
    while len(await_pts) > 0:
        pt = await_pts.pop()
        if not island_map[tuple(pt)] == -1 :
            continue

        isl_idx = isl_idx + 1
        isl_pts.append(pt)
        while len(isl_pts) > 0:
            isl_pt = isl_pts.pop()
            island_map[tuple(isl_pt)] = isl_idx
            
            # fig = plt.figure()
            # fig.add_subplot(1,2,1)
            # plt.imshow(map)
            # fig.add_subplot(1,2,2)
            # plt.imshow(island_map)
            # plt.show()

            py = isl_pt[0]
            px = isl_pt[1]
            if px > 0:
                if not map[py,px-1] == 0 and island_map[py,px-1] == -1:
                    isl_pts.append([py, px-1])

            if py > 0:
                if not map[py-1,px] == 0 and island_map[py-1,px] == -1:
                    isl_pts.append([py-1, px])        

            if px+1 < map.shape[1]:
                if not map[py,px+1] == 0 and island_map[py,px+1] == -1:
                    isl_pts.append([py,px+1])
            
            if py+1 < map.shape[0]:
                if not map[py+1,px] == 0 and island_map[py+1,px] == -1:
                    isl_pts.append([py+1,px])

    return island_map