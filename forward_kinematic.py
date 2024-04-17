import numpy as np
from scipy.spatial.transform import Rotation as R

import file_io as bvh_reader
from viewer import SimpleViewer


def show_T_pose(viewer, joint_names, joint_parents, joint_offsets):
    """
    Show the T-pose of the skeleton
        joint_names:    Shape - (J)     a list to store the name of each joit
        joint_parents:  Shape - (J)     a list to store the parent index of each joint, -1 means no parent
        joint_offsets:  Shape - (J, 1, 3)  an array to store the local offset to the parent joint
        viewer:         SimpleViewer object
        joint_names:    a list to store the name of each joit
        joint_parents:  a list to store the parent index of each joint, -1 means no parent
        joint_offsets:  an array to store the local offset to the parent joint
        joint_rotations:    an array to store the local joint rotation in quaternion representation
    """

    global_joint_position = np.zeros((len(joint_names), 3))
    for joint_idx, parent_idx in enumerate(joint_parents):
        for joint_idx, parent_idx in enumerate(joint_parents):
            if parent_idx == -1:
                global_joint_position[joint_idx] = joint_offsets[joint_idx]
            else:
                global_joint_position[joint_idx] = (
                    global_joint_position[parent_idx] + joint_offsets[joint_idx]
                )
        viewer.set_joint_position_by_name(
            joint_names[joint_idx], global_joint_position[joint_idx]
        )

    viewer.run()


def forward_kinametic(
    viewer,
    joint_names,
    joint_parents,
    joint_offsets,
    joint_positions,
    joint_rotations,
    show_animation=False,
):
    """
    calculate the global joint positions and orientations by FK
    F is Frame number;  J is Joint number
    joint_names:    Shape - (J)     a list to store the name of each joit
    joint_parents:  Shape - (J)     a list to store the parent index of each joint, -1 means no parent
    joint_offsets:  Shape - (J, 1, 3)  an array to store the local offset to the parent joint
    joint_positions:    Shape - (F, J, 3)   an array to store the local joint positions
    joint_rotations:    Shape - (F, J, 4)   an array to store the local joint rotation in quaternion representation
    """
    joint_number = len(joint_names)
    frame_number = joint_rotations.shape[0]

    global_joint_positions = np.zeros((frame_number, joint_number, 3))
    global_joint_orientations = np.zeros((frame_number, joint_number, 4))
    global_joint_orientations[:, :, 3] = 1.0

    joint_number = len(joint_names)
    frame_number = joint_rotations.shape[0]

    global_joint_positions = np.zeros((frame_number, joint_number, 3))
    global_joint_orientations = np.zeros((frame_number, joint_number, 4))
    global_joint_orientations[:, :, 3] = 1.0

    for frame_idx in range(frame_number):
        for joint_idx, parent_idx in enumerate(joint_parents):
            if parent_idx == -1:
                global_joint_positions[frame_idx, joint_idx] = joint_positions[
                    frame_idx, joint_idx
                ]
                global_joint_orientations[frame_idx, joint_idx] = joint_rotations[
                    frame_idx, joint_idx
                ]
            else:
                parent_position = global_joint_positions[frame_idx, parent_idx]
                parent_rotation = R.from_quat(
                    global_joint_orientations[frame_idx, parent_idx]
                )
                offset = joint_offsets[joint_idx]
                rotated_offset = parent_rotation.apply(offset)
                global_joint_positions[frame_idx, joint_idx] = (
                    parent_position + rotated_offset
                )
                global_joint_orientations[frame_idx, joint_idx] = (
                    parent_rotation * R.from_quat(joint_rotations[frame_idx, joint_idx])
                ).as_quat()

        if not show_animation:
            viewer.show_pose(
                joint_names,
                global_joint_positions[frame_idx],
                global_joint_orientations[frame_idx],
            )
        else:
            viewer.show_pose(
                joint_names,
                global_joint_positions[frame_idx],
                global_joint_orientations[frame_idx],
            )

    if not show_animation:
        show_frame_idx = 0
        viewer.show_pose(
            joint_names,
            global_joint_positions[show_frame_idx],
            global_joint_orientations[show_frame_idx],
        )

    else:

        class UpdateHandle:
            def __init__(self):
                self.current_frame = 0

            def update_func(self, viewer_):
                cur_joint_position = global_joint_positions[self.current_frame]
                cur_joint_orentation = global_joint_orientations[self.current_frame]
                viewer.show_pose(joint_names, cur_joint_position, cur_joint_orentation)
                self.current_frame = (self.current_frame + 1) % frame_number

        handle = UpdateHandle()
        viewer.update_func = handle.update_func
    viewer.run()


def main():
    viewer = SimpleViewer()
    bvh_file_path = "data/motion_walking.bvh"

    joint_names, joint_parents, channels, joint_offsets = bvh_reader.load_meta_data(
        bvh_file_path
    )
    _, local_joint_positions, local_joint_rotations = bvh_reader.load_motion_data(
        bvh_file_path
    )

    show_T_pose(viewer, joint_names, joint_parents, joint_offsets)

    forward_kinametic(
        viewer,
        joint_names,
        joint_parents,
        joint_offsets,
        local_joint_positions,
        local_joint_rotations,
        show_animation=True,
    )


if __name__ == "__main__":
    main()
