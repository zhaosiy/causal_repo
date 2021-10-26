from nuscenes import NuScenes

DATAROOT = '/Users/siyanzhao/Desktop/causalcity/causalcity.github.io/code/Nuscenes/v1.0-mini'
nuscenes = NuScenes('v1.0-mini', dataroot=DATAROOT)
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction import PredictHelper
import numpy as np
helper = PredictHelper(nuscenes)
mini_train = get_prediction_challenge_split("mini_train", dataroot=DATAROOT)
print(mini_train[:5], len(mini_train))
instance_token, sample_token = mini_train[0].split("_")
annotation = helper.get_sample_annotation(instance_token, sample_token)
# print(annotation)
future_xy_local = helper.get_future_for_agent(instance_token, sample_token, seconds=2, in_agent_frame=False,
                                              just_xy=True)
# print(future_xy_local)
sample = helper.get_annotations_for_sample(sample_token)

# print(len(sample))
# We get new instance and sample tokens because these methods require computing the difference between records.
instance_token_2, sample_token_2 = mini_train[5].split("_")
# Meters / second.
# print(f"Velocity: {helper.get_velocity_for_agent(instance_token_2, sample_token_2)}\n")
# Meters / second^2.
# print(f"Acceleration: {helper.get_acceleration_for_agent(instance_token_2, sample_token_2)}\n")
from nuscenes.prediction.models.backbone import ResNetBackbone
from nuscenes.prediction.models.covernet import CoverNet
import torch
import matplotlib.pyplot as plt

from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer

'''
static_layer_rasterizer = StaticLayerRasterizer(helper)
agent_rasterizer = AgentBoxesWithFadedHistory(helper, seconds_of_history=1)
mtp_input_representation = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())

instance_token_img, sample_token_img = 'bc38961ca0ac4b14ab90e547ba79fbb6', '7626dde27d604ac28a0240bdd54eba7a'
anns = [ann for ann in nuscenes.sample_annotation if ann['instance_token'] == instance_token_img]
img = mtp_input_representation.make_input_representation(instance_token_img, sample_token_img)

plt.imshow(img)
plt.show()
backbone = ResNetBackbone('resnet50')
# Note that the value of num_modes depends on the size of the lattice used for CoverNet.
covernet = CoverNet(backbone, num_modes=64)
agent_state_vector = torch.Tensor([[helper.get_velocity_for_agent(instance_token_img, sample_token_img),
                                    helper.get_acceleration_for_agent(instance_token_img, sample_token_img),
                                    helper.get_heading_change_rate_for_agent(instance_token_img, sample_token_img)]])

image_tensor = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0)
logits = covernet(image_tensor, agent_state_vector,1)
print(logits)
'''
# nuscenes.list_scenes()
my_scene = nuscenes.scene[5]

first_sample_token = my_scene['first_sample_token']
my_sample = nuscenes.get('sample', first_sample_token)

# create trajectory datasets:
# [num_scene, number_car, 40, 2]
# input:
# [num_scene, number_car, 5, 2(xy)+1(v)+1(a)]
# output:
# [num_scene, number_car, 35, 2] with availability mask

# for each scene, select cars larger than 10 seconds trajectory, select more than 3 cars
# for each car, collect x,y, velocity, acceleration, size

multi_car_data = {}
scene_cnt = 0
for scene_id in range(10):
    this_scene_data = {}
    my_scene = nuscenes.scene[scene_id]
    first_sample_token = my_scene['first_sample_token']
    my_sample = nuscenes.get('sample', first_sample_token)
    all_anns = my_sample['anns']
    car_cnt = 0

    for car_id, k in enumerate(all_anns):
        my_annotation_metadata = nuscenes.get('sample_annotation', k)
        if my_annotation_metadata['category_name'] != 'vehicle.car':
            continue
        ins_token = my_annotation_metadata['instance_token']
        ann_tokens_ = nuscenes.field2token('sample_annotation', 'instance_token', ins_token)
        ann_tokens = set(ann_tokens_)
        if len(ann_tokens) >= 15:
            step = len(ann_tokens)
            availability_mask = np.concatenate([np.ones((1, step)), np.zeros((1, 41-step))], axis=-1)
            this_car_log = {}
            future_local = helper.get_future_for_agent(ins_token, first_sample_token, seconds=20,
                                                       in_agent_frame=True, just_xy=False)
            future_local_xy = helper.get_future_for_agent(ins_token, first_sample_token, seconds=20,
                                                       in_agent_frame=True, just_xy=True)

            acc = np.zeros((40, 1))
            velocity = np.zeros((40, 1))
            trans = np.zeros((40, 2))
            # record current acc, velocity, xy
            cur_velocity = 0
            velocity[0] = cur_velocity
            cur_acc = helper.get_acceleration_for_agent(ins_token, first_sample_token)
            acc[0] = cur_acc
            cur_xy_heading = my_annotation_metadata['translation'][:-1]
            #print(cur_xy_heading,'cur')
            trans[0] = cur_xy_heading
            # record size
            this_car_log['size'] = future_local[0]['size']
            assert this_car_log['size'] == my_annotation_metadata['size']
            # record future logs

            for x in range(1, len(future_local)):
                #print(trans[0],'trans0000000')
                this_translation = future_local[x]['translation']
                this_velocity = helper.get_velocity_for_agent(ins_token, future_local[x]['sample_token'])
                this_acceleration = helper.get_acceleration_for_agent(ins_token, future_local[x]['sample_token'])
                acc[x] = this_acceleration
                velocity[x] = this_velocity
                trans[x] = future_local_xy[x]

            trans[0] = [0, 0]
            assert len(trans) == len(velocity)
            this_car_log['velocity'] = velocity
            this_car_log['trans'] = trans
            #print(trans)
            this_car_log['acc'] = acc
            this_car_log['mask'] = availability_mask
            this_scene_data[str(car_cnt)] = this_car_log
            car_cnt += 1
    if len(this_scene_data) > 3:
        this_scene_data['nbr_car'] = len(this_scene_data)
        multi_car_data[str(scene_cnt)] = this_scene_data
        scene_cnt += 1

for k, v in multi_car_data.items():
        print(k, len(v))
import pickle
with open('motion_data.pkl', 'wb') as output:
    # Pickle dictionary using protocol 0.
    pickle.dump(multi_car_data, output)