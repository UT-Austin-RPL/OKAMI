# mp
import multiprocessing as mp
from okami.plan_generation.utils.misc_utils import *
from okami.plan_generation.utils.tap_utils import *
from okami.plan_generation.algos.hoig import HandObjectInteractionGraph
from okami.plan_generation.algos.tap_segmentation import TAPSegmentation
from okami.plan_generation.algos.temporal_segments import TemporalSegments
from okami.plan_generation.utils.traj_utils import SimpleTrajProcessor

from okami.plan_generation.utils.o3d_utils import *
from okami.plan_generation.utils.log_utils import get_orion_logger
from okami.plan_generation.utils.gpt4v_utils import encode_imgs_from_path, json_parser
from okami.plan_generation.algos.gpt4v import GPT4V

import numpy as np

class HumanVideoPlan:
    def __init__(self):
        self.temporal_segments = TemporalSegments()
        self.hoigs = []
        self.human_video_annotation_path = ""

        self.smplh_traj = np.array([])
        
        self.object_names = []
        self.object_id_to_name = {}

    def generate_from_human_video(self, human_video_annotation_path, zero_pose_name="ready", video_smplh_ratio=1.0, use_smplh=True):
        
        self.human_video_annotation_path = human_video_annotation_path

        tap_segmentation = TAPSegmentation()
        tap_segmentation.load(human_video_annotation_path)
        self.temporal_segments = tap_segmentation.temporal_segments

        self.object_names = get_object_names_from_annotation(human_video_annotation_path)

        retargeter = None
        if use_smplh:
            self.smplh_traj = get_smplh_traj_annotation(human_video_annotation_path)
            len_smplh = len(self.smplh_traj)
            len_video = self.temporal_segments.segments[-1].end_idx
            video_smplh_ratio = len_video / len_smplh
            print("video_smplh_ratio", video_smplh_ratio)
            # retargeter = Retargeter(example_data=self.smplh_traj[0])

        self.hoigs = []
        estimated_extrinsics = None
        for i in range(len(self.temporal_segments.segments)):
            hoig = HandObjectInteractionGraph()
            hoig.create_from_human_video(human_video_annotation_path, 
                                         self.temporal_segments.segments[i].start_idx, 
                                         self.temporal_segments.segments[i].end_idx-1,
                                         segment_idx=i,
                                         extrinsics=estimated_extrinsics,
                                         calibrate_grasp=(i == 0),
                                         zero_pose_name=zero_pose_name,
                                         video_smplh_ratio=video_smplh_ratio,
                                         use_smplh=use_smplh)
            self.hoigs.append(hoig)
            estimated_extrinsics = hoig.camera_extrinsics
        
        self.last_frame = HandObjectInteractionGraph()
        self.last_frame.create_from_human_video(human_video_annotation_path,
                                                self.temporal_segments.segments[-1].end_idx-1,
                                                self.temporal_segments.segments[-1].end_idx,
                                                segment_idx=len(self.temporal_segments.segments),
                                                extrinsics=estimated_extrinsics,
                                                calibrate_grasp=False,
                                                zero_pose_name=zero_pose_name,
                                                video_smplh_ratio=video_smplh_ratio,
                                                use_smplh=use_smplh)

        # get reference object and manipulate object names
        self.get_object_id_to_name()

        self.plan_inference()

        print("generated HOIGs from human video, total_num=", len(self.hoigs), self.temporal_segments)
    
    def get_graph(self, idx) -> HandObjectInteractionGraph:
        return self.hoigs[idx]
    
    @property
    def num_graphs(self):
        return len(self.hoigs)
    
    def get_object_id_to_name(self):
        first_graph = self.get_graph(0)
        for object_id in first_graph.object_ids:
            rgb_img = first_graph.get_objects_2d_image(object_id)
            cv2.imwrite(os.path.join(self.human_video_annotation_path, f"object_{object_id}.png"), rgb_img)
            
            vlm = GPT4V()
            vlm.begin_new_dialog()
            text_description = f'''Please Identify the object in the image below. You need to choose from the following objects: {self.object_names}.''' + '''
Your output format:
```json
{
    "object_name": "OBJECT_NAME",
}
```

Ensure the response can be parsed by Python `json.loads`, e.g.: no trailing commas, no single quotes, etc.
You should output the name of the object in a string "OBJECT_NAME" that can be easily parsed by Python. The name is a string, e.g., "apple", "pen", "keyboard", etc.
'''
            text_prompt_list = [text_description]
            base64_img_list = [vlm.preprocess_image(rgb_img, is_rgb=True, encoding='jpg')]
            text_response = vlm.run(text_prompt_list, base64_img_list)
            json_data = json_parser(text_response)

            if json_data is not None and 'object_name' in json_data:
                self.object_id_to_name[object_id] = json_data["object_name"]
            
        print("object_id_to_name", self.object_id_to_name)
        print()

    def heuristic_get_segment_type(self, graph_id):
        
        current_graph = self.get_graph(graph_id)

        # last graph cannot be reaching for objects
        if graph_id == self.num_graphs - 1:
            # For the last step, if there is no hand-object contact, then it is 'release' step.
            if current_graph.arms_contact[0] == -1 and current_graph.arms_contact[1] == -1:
                return 'release'
            return 'manipulation'

        next_graph = self.get_graph(graph_id + 1)

        # If there exists a hand that contact a new object in next graph, then this step is 'reach'.
        for arm_idx in range(2):
            if next_graph.arms_contact[arm_idx] >= 0:
                if current_graph.arms_contact[arm_idx] < 0:
                    return 'reach'
                if current_graph.arms_contact[arm_idx] != next_graph.arms_contact[arm_idx]: # Note: this circumstance most likely won't happen
                    return 'reach'
        
        if current_graph.arms_contact[0] == -1 and current_graph.arms_contact[1] == -1:
            return 'release'
        
        return 'manipulation'


    def vlm_get_segment_type(self, graph_id):
        current_graph = self.get_graph(graph_id)
        
        vlm = GPT4V()
        vlm.begin_new_dialog()
        text_description = '''The following images shows a human motion, where the human is either reaching for some objects, or is manipulating some objects that is already in hand. 
More specifically, 'reaching' involves moving the arm to touch or grasp an object, while 'manipulating' involves handling or controlling the object after contact.
Your job is to first describe what human is doing, then classify the type of this motion to either 'reach' or 'manipulation' based on the above definition. 

Your output format is:

Description: The human is xxx.
The type of the motion is:
```json
{
    "type": "reach" or "manipulate",
}
```

Ensure the response can be parsed by Python `json.loads`, e.g.: no trailing commas, no single quotes, etc.
Ensure that the type you output is either 'reach' or 'manipulate', do not output anything else.
'''
        text_prompt_list = [text_description]
        base64_img_list = [[vlm.preprocess_image(img, is_rgb=True, encoding='jpg') for img in current_graph.representative_images]]

        text_response = vlm.run(text_prompt_list, base64_img_list)
        json_data = json_parser(text_response)

        print("response =", text_response)
        print("determined type=", json_data['type'])
        print()
        return json_data['type']

    def vlm_get_reference_object(self, graph_id):
        current_graph = self.get_graph(graph_id)

        manipulate_object_name = self.object_id_to_name[current_graph.get_manipulate_object_id()[0]]

        vlm = GPT4V()
        vlm.begin_new_dialog()
        text_description = f'''The following images shows a manipulation motion, where the human is manipulating the object {manipulate_object_name}.
Please identify the reference object in the image below, which could be an object on which to place {manipulate_object_name}, or an object which {manipulate_object_name} is interacting with.
Note that there may not necessarily have an reference object, as sometimes human may just interacting with the object itself, like throwing it, or spinning it around.
You need to first identify if there is a reference object. If there is no reference object, you should output 'None'. If there exists a reference object, you need to output the reference object's name chosen from the following objects: {self.object_names}.''' + '''
Your output format is:

```json
{
    "reference_object_name": "REFERENCE_OBJECT_NAME" or None,
}
```

Ensure the response can be parsed by Python `json.loads`, e.g.: no trailing commas, no single quotes, etc.
'''
        text_prompt_list = [text_description]
        base64_img_list = [[vlm.preprocess_image(img, is_rgb=True, encoding='jpg') for img in current_graph.representative_images]]

        for try_idx in range(5):
            try:
                text_response = vlm.run(text_prompt_list, base64_img_list)
                json_data = json_parser(text_response)
                break
            except Exception as e:
                print("error", e)
                print("retrying...")
                json_data = {'reference_object_name': 'None'}
                continue

        if json_data['reference_object_name'] == 'None':
            current_graph.set_reference_object_id(-1)
        else:
            for object_id, object_name in self.object_id_to_name.items():
                if object_name == json_data['reference_object_name']:
                    if object_name == manipulate_object_name:
                        current_graph.set_reference_object_id(-1)
                    else:
                        current_graph.set_reference_object_id(object_id)
                    break
        print("reference object id", current_graph.get_reference_object_id(), 'name', json_data['reference_object_name'])
        print()

    def vlm_get_manipulate_object(self, graph_id):
        current_graph = self.get_graph(graph_id)
        
        vlm = GPT4V()
        vlm.begin_new_dialog()

        text_description = f'''The following images shows a manipulation motion, where the human is manipulating an object. Your task is to determine which object is being manipulated in the images below. You need to choose from the following objects: {self.object_names}.
Tips: the manipulated object is the object that the human is interacting with, such as picking up, moving, or pressing, and it is in contact with human's {current_graph.moving_arm} hand.''' + '''
Your output format is:

```json
{
    "manipulate_object_name": "MANIPULATE_OBJECT_NAME",
}
```

Ensure the response can be parsed by Python `json.loads`, e.g.: no trailing commas, no single quotes, etc.
'''
        text_prompt_list = [text_description]
        base64_img_list = [[vlm.preprocess_image(img, is_rgb=True, encoding='jpg') for img in current_graph.representative_images]]

        text_response = vlm.run(text_prompt_list, base64_img_list)
        json_data = json_parser(text_response)

        for object_id, object_name in self.object_id_to_name.items():
            if object_name == json_data['manipulate_object_name']:
                current_graph.set_manipulate_object_id([object_id])
                break
        print("manipulate object id", current_graph.get_manipulate_object_id(), 'name', json_data['manipulate_object_name'])


    def plan_inference(self, velocity_threshold=0.8, target_dist_threshold=0.05, target_intersection_threshold=600):
        """This is a function to infer the important information in a plan. Specifically, we will get the following information from this function: 
            1. Manipulate object sequence, which specifies the object that the robot should manipulate in each step
            2. Reference object sequence, which specifies the object that the robot should use as reference in each step

            -1 in the object sequence means that no object is selected for manipulation or reference.
        Args:
            velocity_threshold (_type_, optional): _description_. Defaults to 1..

        Returns:
            _type_: _description_
        """
        for graph_id in range(self.num_graphs):

            graph_in_query = self.get_graph(graph_id)
            assert(type(graph_in_query) == HandObjectInteractionGraph)

            # determine segment type
            segment_type = self.heuristic_get_segment_type(graph_id)
            graph_in_query.set_segment_type(segment_type)
            print("segment_type=", segment_type, graph_in_query.arms_contact)
            print()

            # the object with the highest velocity is selected as the object to manipulate
            candidate_objects_to_move = []
            v_mean_list = []
            v_std_list = []
            for object_id in graph_in_query.object_ids:
                point_nodes = graph_in_query.select_point_nodes(object_ids=[object_id])
                points = graph_in_query.get_point_list(point_nodes=point_nodes)
                world_trajs = graph_in_query.get_pixel_trajs(object_ids=[object_id])
                all_visibilities = graph_in_query.get_visibility_trajs(object_ids=[object_id])
                traj_diff = world_trajs[:, 1:, :] - world_trajs[:, :-1, :]
                confidence = all_visibilities[:, 1:] * all_visibilities[:, :-1]

                traj_diff = traj_diff * confidence[:, :, None]

                v_mean = np.mean(np.linalg.norm(traj_diff, axis=-1))
                v_std = np.std(np.linalg.norm(traj_diff, axis=-1))

                if v_mean > velocity_threshold:
                    v_mean_list.append(v_mean)
                    v_std_list.append(v_std)
                    candidate_objects_to_move.append(object_id)

                print("velocity of object", object_id, ":", v_mean, v_std)

            if segment_type == 'reach':
                # For reaching segment, the manipulate objects are the objects that gain contact with hands in next step
                assert(graph_id < self.num_graphs - 1)
                next_graph = self.get_graph(graph_id + 1)
                
                manipulate_object_candidate = []
                for arm_idx in range(2):
                    if next_graph.arms_contact[arm_idx] >= 0 and graph_in_query.arms_contact[arm_idx] < 0:
                        if next_graph.arms_contact[arm_idx] + 1 not in manipulate_object_candidate:
                            manipulate_object_candidate.append(next_graph.arms_contact[arm_idx] + 1)
                assert(len(manipulate_object_candidate) > 0)

                graph_in_query.set_manipulate_object_id(manipulate_object_candidate)
            elif segment_type == 'manipulation':
                # For manipulation segment, the manipulate object is the object with the highest velocity (there is only one manipulate object).
                if len(candidate_objects_to_move) == 0:
                    print("No moving objects found. Will try to determine the object being manipulated using vlm.")
                    self.vlm_get_manipulate_object(graph_id)
                else:
                    if (len(candidate_objects_to_move) == 1) or (np.std(v_mean_list) < 0.1):
                        graph_in_query.set_manipulate_object_id([candidate_objects_to_move[0]])
                    else:
                        # sort candidate objects to move by v_mean
                        candidate_objects_to_move = [x for _, x in sorted(zip(v_mean_list, candidate_objects_to_move))]
                        graph_in_query.set_manipulate_object_id([candidate_objects_to_move[-1]])
            else:
                # release step
                graph_in_query.set_manipulate_object_id([-1])

            # Get point clouds for all objects
            pcd_list = []
            for object_id in graph_in_query.object_ids:
                pcd_array, _ = graph_in_query.get_objects_3d_points(object_id=object_id, filter=False, downsample=False)
                pcd = create_o3d_from_points_and_color(pcd_array)
                pcd_list.append(pcd)

            dists = []
            contact_states = []

            dist_threshold = target_dist_threshold
            intersection_threshold = target_intersection_threshold
            max_intersection = 0

            dists = []
            contact_states = []

            objects_in_contact = []
            for i in range(len(pcd_list)):
                for j in range(i+1, len(pcd_list)):
                    dists = pcd_list[i].compute_point_cloud_distance(pcd_list[j])
                    intersections = [d for d in dists if d < dist_threshold]
                    # print(i+1, ", ", j+1, " | ", len(intersections))
                    if len(intersections) > intersection_threshold:
                        contact_states.append((i+1, j+1))
                        objects_in_contact.append(i+1)
                        objects_in_contact.append(j+1)

            objects_in_contact = list(set(objects_in_contact))
            if graph_id > 0:
                prev_graph = self.get_graph(graph_id - 1)
                if len(prev_graph.contact_states) > 0:
                    for contact_state in prev_graph.contact_states:
                        if contact_state[0] not in objects_in_contact and contact_state[1] not in objects_in_contact:
                            contact_states.append(contact_state)

            graph_in_query.contact_states = contact_states

            print("manipulate object id", graph_in_query.get_manipulate_object_id())
            print("contact states", contact_states)
            print()

            rgb_imgs = [
                graph_in_query.get_objects_2d_image(obj_id) for obj_id in graph_in_query.get_manipulate_object_id()
            ]
            os.makedirs(os.path.join(self.human_video_annotation_path, "objects"), exist_ok=True)
            for j in range(len(rgb_imgs)):
                cv2.imwrite(os.path.join(self.human_video_annotation_path, "objects", f"manipulate_object_{graph_id}_{j}.png"), rgb_imgs[j])
            # cv2.imwrite(os.path.join(self.human_video_annotation_path, "objects", f"manipulate_object_{graph_id}.png"), rgb_img)

        # decide reference objects between a pair of graphs
        for graph_id in range(self.num_graphs):
            current_graph = self.get_graph(graph_id)
            current_graph.set_reference_object_id(-1)

            # only need to determine reference object for manipulation segment
            if current_graph.get_segment_type() == 'manipulation':
                assert(current_graph.get_manipulate_object_id()[0] > 0)

                if graph_id < self.num_graphs - 1:
                    # print(f"graph_id={graph_id}, num_graphs={self.num_graphs}")
                    next_graph = self.get_graph(graph_id + 1)
                    contact_states = next_graph.contact_states
                    manipulate_object_id = current_graph.get_manipulate_object_id()[0]
                else:
                    contact_states = []

                if len(contact_states) > 0:
                    for contact_state in contact_states:
                        if manipulate_object_id in contact_state:
                            temp_contact_state = list(contact_state)
                            temp_contact_state.remove(manipulate_object_id)
                            current_graph.set_reference_object_id(temp_contact_state[0])
                            # print("set reference object id", temp_contact_state[0])
                            break
                
                if current_graph.get_reference_object_id() < 0:
                    print("cannot determine reference object based on contacts, but there are likely to have at most one since there exists manipulate object. Trying to call vlm")
                    self.vlm_get_reference_object(graph_id)

            print()
            print("for step", graph_id, "type:", current_graph.get_segment_type(), "contact:", current_graph.arms_contact)
            print("manipulate object id", current_graph.get_manipulate_object_id())
            print("reference object id", current_graph.get_reference_object_id())
            print()

            rgb_img = current_graph.get_objects_2d_image(current_graph.get_reference_object_id())
            cv2.imwrite(os.path.join(self.human_video_annotation_path, "objects", f"reference_object_{graph_id}.png"), rgb_img)

            # calculate needed translation for each segment

            if current_graph.get_segment_type() == 'manipulation':
                # obtain the translation between manipulate object and reference object
                if current_graph.get_manipulate_object_id()[0] > 0 and current_graph.get_reference_object_id() > 0:
                    
                    if graph_id + 1 < self.num_graphs:
                        next_graph = self.get_graph(graph_id + 1)
                    else:
                        next_graph = self.last_frame

                    mani_id = current_graph.get_manipulate_object_id()[0]
                    ref_id = current_graph.get_reference_object_id()

                    pcd_manipulate, _ = next_graph.get_objects_3d_points(object_id=mani_id, filter=False, downsample=False)
                    pcd_reference, _ = next_graph.get_objects_3d_points(object_id=ref_id, filter=False, downsample=False)

                    manipulate_center = np.mean(np.array(pcd_manipulate), axis=0)
                    reference_center = np.mean(np.array(pcd_reference), axis=0)
                    translation = manipulate_center - reference_center

                    current_graph.target_translation_btw_objects = translation
                    print("translation", translation)

            elif current_graph.get_segment_type() == 'reach':
                
                next_graph = self.get_graph(graph_id + 1)
                
                for arm_idx in range(2):
                    if next_graph.arms_contact[arm_idx] >= 0 and current_graph.arms_contact[arm_idx] < 0:
                        mani_id = next_graph.arms_contact[arm_idx] + 1
                        
                        # obtain the translation between hand and object
                        
                        pcd_manipulate, _ = next_graph.get_objects_3d_points(object_id=mani_id, filter=False, downsample=False)
                        object_center = np.mean(np.array(pcd_manipulate), axis=0)

                        hand_center = next_graph.estimate_hand_interaction_center[arm_idx]
                        palm = next_graph.estimate_palm_point[arm_idx]

                        # translation = hand_center - object_center
                        translation = palm - object_center
                        
                        current_graph.target_hand_object_translation[arm_idx] = translation
                        print(f"arm{arm_idx} translation", translation)

        # check all graphs and see if things are feasible: the manipulate object should be in contact with the hand, if not, then the segment should be categorized as 'release'
        for graph_id in range(self.num_graphs):
            current_graph = self.get_graph(graph_id)
            if current_graph.get_segment_type() == 'manipulation':
                if current_graph.get_manipulate_object_id()[0] > 0:
                    manipulate_object_id = current_graph.get_manipulate_object_id()[0]
                    if current_graph.arms_contact[0] + 1 != manipulate_object_id and \
                        current_graph.arms_contact[1] + 1 != manipulate_object_id:
                        current_graph.set_segment_type('release')
                        current_graph.set_manipulate_object_id([-1])
                        current_graph.set_reference_object_id(-1)
                        current_graph.target_translation_btw_objects = np.array([])

        return
    
    def get_num_segments(self):
        return len(self.hoigs)
    
    def get_all_segments_info(self):
        # TODO: update info
        segments = []
        for idx, segment in enumerate(self.temporal_segments.segments):
            segments.append(convert_to_json_serializable({
                'start_idx': segment.start_idx,
                'end_idx': segment.end_idx,
                'segment_type': self.hoigs[idx].segment_type,
                'hand_contact': [(self.object_id_to_name[obj_id + 1] if obj_id >= 0 else "None")
                                  for obj_id in self.hoigs[idx].arms_contact],
                'moving_arm': self.hoigs[idx].moving_arm,
                'manipulate_object': [(self.object_id_to_name[obj_id] if obj_id > 0 else "None") 
                                       for obj_id in self.hoigs[idx].manipulate_object_id],
                'reference_object': self.object_id_to_name[self.hoigs[idx].reference_object_id] if self.hoigs[idx].reference_object_id > 0 else "None",
                'target_object_translation': self.hoigs[idx].target_translation_btw_objects.tolist(),
                'target_hand_object_translation': [t.tolist() for t in self.hoigs[idx].target_hand_object_translation],
                # 'smplh_traj': convert_to_json_serializable(self.hoigs[idx].smplh_traj),
                # 'objects': self.object_names
            }))
            # print(segments[-1])
        return segments
    
    def visualize_plan(self, no_smplh=False):
        video_seq = get_video_seq_from_annotation(self.human_video_annotation_path)

        initial_images = []
        for segment in self.temporal_segments.segments:
            initial_images.append(video_seq[segment.start_idx])
        video_writer = VideoWriter(self.human_video_annotation_path, video_name=f"plan_vis.mp4", fps=30)
        for i in range(len(video_seq)):
            image_frame = []
            for idx, segment in enumerate(self.temporal_segments.segments):
                if segment.in_duration(i):
                    img = video_seq[i].copy()
                elif i >= segment.end_idx:
                    img = video_seq[segment.end_idx].copy()
                else:
                    img = video_seq[segment.start_idx].copy()
                image_frame.append(img)
            
            image_frame = np.concatenate(image_frame, axis=1)
            video_writer.append_image(image_frame)
        video_writer.save(bgr=False)
    
    def add_text_on_image(self, img, text, pos=(10,30), color=(255,255,255), fontsize=1):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = fontsize
        fontColor = color
        lineType = 2
        cv2.putText(img, text, 
            pos, 
            font, 
            fontScale,
            fontColor,
            lineType)
        return img
