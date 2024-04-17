import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from file_io import BVHMotion
from Viewer.controller import SimpleViewer
from Viewer.viewer import ShowBVHUpdate


def interpolation(left_data, right_data, t, method="linear", return_first_key=True):
    res = [left_data] if return_first_key else []
    """
    This function performs interpolation for position and rotation data using linear interpolation for position and Slerp interpolation for rotation. 
    The input data should have a shape of (num_joints, 3) for position and (num_joints, 4) for rotation, where the rotation data is in quaternion format.

    For linear interpolation of position, the function uses the following formula:
        data_between = left_data + (right_data - left_data) * (i / t)
    where left_data and right_data are the position data of the adjacent frames, i is the current frame index, and t is the total number of frames.

    For Slerp interpolation of rotation, the function utilizes the scipy.spatial.transform.Slerp class. 
    The Slerp interpolation can be achieved with the following steps:

    1. Convert the quaternion rotation data into the scipy.spatial.transform.Rotation format using R.from_quat([q1, q2]), where q1 and q2 are the quaternions of the adjacent frames.
    2. Define key frames as [0, 1].
    3. Create an instance of Slerp using slerp = Slerp(key_times, key_rots), where key_times represents the key frame times and key_rots is the rotation data in the scipy.spatial.transform.Rotation format.
    4. Generate new key frames using np.linspace(0, 1, t+1).
    5. Interpolate the rotations at the new key frames using interp_rots = slerp(new_key_frames).
    6. Iterate over each joint in a loop with joint_idx and extract the quaternion for that joint from left_data, combining them after the loop.
    7. Exclude the last frame of the interpolated rotations (interp_rots) as it is identical to right_data.
    
    Note: 
        The Slerp interpolation in scipy does not directly support quaternion data with a shape of (num_joints, 4). 
        Therefore, the function requires looping over each joint (joint_num) and extracting the quaternion for each joint individually.    
    """
    if method == "linear":
        for i in range(1, t + 1):
            data_between = left_data + (right_data - left_data) * (i / t)
            res.append(data_between)
        return res
    elif method == "slerp":
        num_joints = left_data.shape[0]
        key_frames = np.array([0, 1])
        interp_rots = np.zeros((t, num_joints, 4))
        for joint_idx in range(num_joints):
            key_rots = R.from_quat(
                np.vstack([left_data[joint_idx], right_data[joint_idx]])
            )
            slerp = Slerp(key_frames, key_rots)
            new_key_frames = np.linspace(0, 1, t + 1)
            interp_rots[:, joint_idx] = slerp(new_key_frames[1:]).as_quat()

        res.extend(interp_rots)
        return res


def key_framing(viewer, time_step, target_step):
    motion = BVHMotion("data/motion_walking.bvh")

    motio_length = motion.local_joint_positions.shape[0]
    keyframes = np.arange(0, motio_length, time_step)

    new_motion_local_positions, new_motion_local_rotations = [], []

    previous_frame_idx = 0
    for current_frame_idx in keyframes[1:]:
        between_local_pos = interpolation(
            motion.local_joint_positions[previous_frame_idx],
            motion.local_joint_positions[current_frame_idx],
            target_step - 1,
            "linear",
        )
        between_local_rot = interpolation(
            motion.local_joint_rotations[previous_frame_idx],
            motion.local_joint_rotations[current_frame_idx],
            target_step - 1,
            "slerp",
        )
        new_motion_local_positions.append(between_local_pos)
        new_motion_local_rotations.append(between_local_rot)
        previous_frame_idx = current_frame_idx

    res_motion = motion.raw_copy()
    res_motion.local_joint_positions = np.concatenate(new_motion_local_positions)
    res_motion.local_joint_rotations = np.concatenate(new_motion_local_rotations)

    translation, orientation = res_motion.batch_forward_kinematics()
    task = ShowBVHUpdate(viewer, res_motion.joint_name, translation, orientation)
    viewer.addTask(task.update)


def concatenate_two_motions(
    motion1,
    motion2,
    last_frame_index,
    start_frame_index,
    between_frames,
    searching_frames=20,
    method="interpolation",
):
    """
    This function performs concatenation of two motions, motion1 and motion2, by interpolating the local joint positions and rotations. The inputs should have the following shapes:

    motion1.local_joint_positions: (num_frames, num_joints, 3)
    motion1.local_joint_rotations: (num_frames, num_joints, 4)
    motion2.local_joint_positions: (num_frames, num_joints, 3)
    motion2.local_joint_rotations: (num_frames, num_joints, 4)
    The concatenation process involves five steps:

    Step 1: Get the searching windows for motion1 and motion2
        Define the searching window for motion1 as win_1 = motion1.local_joint_rotations[last_frame_index - searching_frames:last_frame_index + searching_frames]
        Define the searching window for motion2 as win_2 = motion2.local_joint_rotations[max(0, start_frame_indx - searching_frames):start_frame_indx + searching_frames]
    Step 2: Find the closest frame in the searching windows
        Use the similarity matrix in Dynamic Time Warping (DTW) to find the closest frame. The similarity matrix sim_matrix has a shape of (win_1.shape[0], win_2.shape[0]).
        Each element in sim_matrix is calculated as sim_matrix[i, j] = np.linalg.norm(search_source[i] - search_target[j]).
        Find the minimum value in sim_matrix and get the corresponding indices i and j.
    Step 3: Convert the indices from the searching window to the indices in the original motion sequence.
    Step 4: Perform interpolation between motion1 and motion2
        Obtain the pose in motion1 at index real_i and the pose in motion2 at index real_j.
        Perform interpolation for both positions and rotations.
        Note: Before interpolation, align the root positions of motion2 (at real_j) to the root positions of motion1 (at real_i) by
        updating the variable motion2.local_joint_positions = motion2.local_joint_positions - ? (the shifting value is not specified here).
    Step 5: Combine motion1, the interpolated frames, and motion2 into one motion sequence.

    Note: The exact details of the interpolation and combining process are not provided here and should be implemented separately.
    """
    win_1 = motion1.local_joint_rotations[
        last_frame_index - searching_frames : last_frame_index + searching_frames
    ]
    win_2 = motion2.local_joint_rotations[
        max(0, start_frame_index - searching_frames) : start_frame_index
        + searching_frames
    ]

    sim_matrix = np.zeros((win_1.shape[0], win_2.shape[0]))
    for i in range(win_1.shape[0]):
        for j in range(win_2.shape[0]):
            sim_matrix[i, j] = np.linalg.norm(win_1[i] - win_2[j])

    min_idx = np.unravel_index(np.argmin(sim_matrix), sim_matrix.shape)
    i, j = min_idx[0], min_idx[1]

    real_i = last_frame_index - searching_frames + i
    real_j = max(0, start_frame_index - searching_frames) + j

    motion1_velocities = np.diff(motion1.local_joint_positions, axis=0)
    motion2_velocities = np.diff(motion2.local_joint_positions, axis=0)

    motion1_weights = np.linalg.norm(motion1_velocities, axis=1)
    motion2_weights = np.linalg.norm(motion2_velocities, axis=1)

    motion1_weights /= np.sum(motion1_weights)
    motion2_weights /= np.sum(motion2_weights)

    weighted_velocities = (
        motion1.local_joint_positions[real_i]
        + (
            motion2.local_joint_positions[real_j]
            - motion1.local_joint_positions[real_i]
        )
        * motion2_weights[0]
    )

    motion2.local_joint_positions = (
        motion2.local_joint_positions
        - motion2.local_joint_positions[0]
        + weighted_velocities
    )

    between_local_pos = interpolation(
        motion1.local_joint_positions[real_i],
        motion2.local_joint_positions[real_j],
        between_frames,
        "linear",
    )
    between_local_rot = interpolation(
        motion1.local_joint_rotations[real_i],
        motion2.local_joint_rotations[real_j],
        between_frames,
        "slerp",
    )

    res = motion1.raw_copy()
    res.local_joint_positions = np.concatenate(
        [
            motion1.local_joint_positions[:real_i],
            between_local_pos,
            motion2.local_joint_positions[real_j:],
        ],
        axis=0,
    )
    res.local_joint_rotations = np.concatenate(
        [
            motion1.local_joint_rotations[:real_i],
            between_local_rot,
            motion2.local_joint_rotations[real_j:],
        ],
        axis=0,
    )
    return res


def concatenate(viewer, between_frames, do_interp=True):
    walk_forward = BVHMotion("data/motion_walking.bvh")
    run_forward = BVHMotion("data/motion_running.bvh")
    run_forward.adjust_joint_name(walk_forward.joint_name)

    last_frame_index = 40
    start_frame_indx = 0

    if do_interp:
        motion = concatenate_two_motions(
            walk_forward,
            run_forward,
            last_frame_index,
            start_frame_indx,
            between_frames,
            method="interpolation",
        )
    else:
        motion = walk_forward.raw_copy()
        motion.local_joint_positions = np.concatenate(
            [
                walk_forward.local_joint_positions[:last_frame_index],
                run_forward.local_joint_positions[start_frame_indx:],
            ],
            axis=0,
        )
        motion.local_joint_rotations = np.concatenate(
            [
                walk_forward.local_joint_rotations[:last_frame_index],
                run_forward.local_joint_rotations[start_frame_indx:],
            ],
            axis=0,
        )

    translation, orientation = motion.batch_forward_kinematics()
    task = ShowBVHUpdate(viewer, motion.joint_name, translation, orientation)
    viewer.addTask(task.update)
    pass


def main():
    viewer = SimpleViewer()

    # key_framing(viewer, 10, 10)
    # key_framing(viewer, 10, 5)
    # key_framing(viewer, 10, 20)
    # key_framing(viewer, 10, 30)
    # key_framing(viewer, 20, 100)
    # concatenate(viewer, between_frames=8, do_interp=False)
    concatenate(viewer, between_frames=15)
    viewer.run()


if __name__ == "__main__":
    main()
