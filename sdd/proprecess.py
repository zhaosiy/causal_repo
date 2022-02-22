import os
import glob
import pickle
import numpy as np

originalfolder = 'annotations/'
subfolder = []
for i in range(len(glob.glob(originalfolder + "*"))):
    subfolder.append(os.path.basename(sorted(glob.glob(originalfolder + "*"))[i]))
classes = ["Pedestrian", "Biker", "Skater", "Car"]
scenes = subfolder[0]
NUM_SCENES = 5
TEST_SCENE = 6
scene_data = {}
runid = 0
for s in range(NUM_SCENES):
    # load annotations
    dataset = open(originalfolder + 'deathCircle/video' + str(s) + '/annotations.txt')
    # dictionary to hold parsed details
    scene = {}

    while True:
        line = dataset.readline()
        #print(line)
        if line == '':
            break
        row = line.split(" ")
        frame = int(row[5])
        if frame % 12 != 0:
            continue
        if int(row[6]) != 0 or int(row[7]) != 0:
            continue
        x = (int(row[1]) + int(row[3])) / 2
        y = (int(row[2]) + int(row[4])) / 2
        label = row[-1][1:-2]
        # skip sparse busses and resolve cars as carts
        if label == "Bus":
            continue
        member_id = int(row[0])

        info = [member_id, (x, y), label]
        if frame in scene:
            scene[frame].append(info)
        else:
            scene[frame] = [info]
    outlay_dict, class_dict, path_dict = {}, {}, {}
    frames = scene.keys()
    frames = sorted(frames)
    # print(len(frames), len(frames) / 2.5 / 60, s)
    window_size = 20  # 20 frames 8 in 12 out
    input_xy = {}
    output_xy = {}
    for frame in frames:
        outlay_dict[frame], path_dict[frame] = {}, {}
        for obj in scene[frame]:
            obj_id = obj[0]
            obj_xy = obj[1]
            outlay_dict[frame][obj_id] = obj_xy  # xy
            class_dict[obj_id] = obj[2]

            if frame == 0:
                path_dict[frame][obj_id] = [obj_xy]
                continue

            prev_frame = frames[frames.index(frame) - 1]
            if obj_id not in path_dict[prev_frame]:
                path_dict[frame][obj_id] = [obj_xy]
            else:
                path_dict[frame][obj_id] = path_dict[prev_frame][obj_id] + [obj[1]]
                # print(len(path_dict[frame][obj_id]))

    for window in range(0, len(frames), 20):
        cur_frame = frames[window]
        initial_num_objects = len(outlay_dict[cur_frame])
        #print('number of objects at this frame', initial_num_objects)
        # constructing a simpler dataset for naive training
        if initial_num_objects > 10:
            objs = outlay_dict[cur_frame]
            #print(len(objs),'number objects')
            this_window_paths = []
            for obj in objs:
                # get future trajectories within 20 frames
                future_frame = window + 20
                if future_frame < len(frames):
                    if frames[future_frame] < frames[-1]:
                        if obj in path_dict[frames[future_frame]]:
                            #print(len(path_dict[frames[future_frame]][obj]),'path thus far')
                            if len(path_dict[frames[future_frame]][obj]) >= 20:
                                obj_path = path_dict[frames[future_frame]][obj][-20:]
                                #print(len(obj_path),'obj path')
                                this_window_paths.append(obj_path)
            #print(len(this_window_paths),'this window number paths',window,future_frame,len(frames),frames[future_frame],frames[-1])
            # number of agents bigger than 5 to make interaction make sense.
            if len(this_window_paths) > 5:
                this_window_trajs = np.zeros((len(this_window_paths), 20, 2))
                #print(this_window_trajs.shape)
                for idx, i in enumerate(this_window_paths):
                    this_window_trajs[idx] = i
                scene_data[str(runid)] = this_window_trajs
                runid += 1

for k, v in scene_data.items():
        print(k, v.shape)
import pickle
with open('SDD_motion_data.pkl', 'wb') as output:
    # Pickle dictionary using protocol 0.
    pickle.dump(scene_data, output)
