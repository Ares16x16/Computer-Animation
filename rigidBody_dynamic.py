import taichi as ti
import numpy as np

# Set up Taichi
ti.init(arch=ti.cpu, debug=True)


# Function to read an OBJ file
def read_obj_file(file_path, scale=1.0):
    vertices = []
    faces = []

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            elements = line.split()
            if elements[0] == "v":
                vertices.append([scale * float(e) for e in elements[1:]])
            elif elements[0] == "f":
                faces.append([int(e.split("/")[0]) - 1 for e in elements[1:]])

    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)


# Read an OBJ file, feel free to change the file path to use your own mesh
file_path = "cat.obj"
vertices_np, faces_np = read_obj_file(file_path, scale=0.3)

faces = ti.field(dtype=ti.i32, shape=faces_np.shape)
# Particle state
particle_vertices = ti.Vector.field(3, dtype=ti.f32, shape=vertices_np.shape[0])
particle_origin_vertices = ti.Vector.field(3, dtype=ti.f32, shape=vertices_np.shape[0])
particle_velocities = ti.Vector.field(3, dtype=ti.f32, shape=vertices_np.shape[0])
particle_force = ti.Vector.field(3, dtype=float, shape=vertices_np.shape[0])

# Body state
body_cm_position = ti.Vector.field(3, dtype=float, shape=())
body_origin_cm_position = ti.Vector.field(3, dtype=float, shape=())
body_velocity = ti.Vector.field(3, dtype=float, shape=())
body_angular_velocity = ti.Vector.field(3, dtype=float, shape=())
body_rotation = ti.Matrix.field(3, 3, dtype=float, shape=())
body_rotation_quaternion = ti.Vector.field(4, dtype=float, shape=())
body_angular_momentum = ti.Vector.field(3, dtype=float, shape=())
body_origin_inverse_inertia = ti.Matrix.field(3, 3, dtype=float, shape=())
body_mass = ti.field(float, shape=())

# Simulation parameters, feel free to change them
# assume all particles have the same mass
particle_mass = 1
initial_velocity = ti.Vector([3.0, 0.0, 0.0])
initial_angular_velocity = ti.Vector([0.0, 0.0, 0.0])
gravity = ti.Vector([0.0, -9.8, 0.0])
# stiffness of the collision
collision_stiffness = 1e4
velocity_damping_stiffness = 1e4
friction_stiffness = 3
# simulation integration time step
dt = 1e-3

# Initialize the fields
# Copy the vertices and faces numpy data to Taichi Fields
particle_vertices.from_numpy(vertices_np)
particle_origin_vertices.from_numpy(vertices_np)
faces.from_numpy(faces_np)

# Indices field for rendering
indices = ti.field(int, shape=3 * faces_np.shape[0])
for i in range(faces_np.shape[0]):
    indices[3 * i] = faces[i, 0]
    indices[3 * i + 1] = faces[i, 1]
    indices[3 * i + 2] = faces[i, 2]


@ti.kernel
def initial():
    # Initialize the body and particle state
    # Compute the center of mass and mass of the body
    for i in ti.grouped(particle_vertices):
        body_mass[None] += particle_mass
        body_cm_position[None] += particle_mass * particle_vertices[i]
    body_cm_position[None] /= body_mass[None]
    body_origin_cm_position[None] = body_cm_position[None]

    # Compute the inertia of the body
    inertia = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    # Compute the inertia tensor of the body
    for i in ti.grouped(particle_vertices):
        r = particle_vertices[i] - body_cm_position[None]
        r_outer = ti.Matrix.outer_product(r, r)
        inertia += particle_mass * (
            r.norm_sqr() * ti.Matrix.identity(float, 3) - r_outer
        )

    # Compute the inverse inertia of the body and store it in the field
    body_origin_inverse_inertia[None] = inertia.inverse()

    # Initialize the particle velocities
    for i in ti.grouped(particle_vertices):
        particle_velocities[i] = initial_velocity

    # Initialize the body state
    body_velocity[None] = initial_velocity
    body_angular_velocity[None] = initial_angular_velocity
    body_angular_momentum[None] = inertia @ initial_angular_velocity

    # Initialize the rotation matrix and quaternion
    body_rotation[None] = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    body_rotation_quaternion[None] = ti.Vector([1.0, 0.0, 0.0, 0.0])


initial()


# quaternion multiplication, this is used to update the rotation quaternion
@ti.func
def quaternion_multiplication(p: ti.template(), q: ti.template()) -> ti.template():
    return ti.Vector(
        [
            p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
            p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
            p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
            p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0],
        ]
    )


# quaternion to rotation matrix
@ti.func
def quaternion_to_matrix(q: ti.template()) -> ti.template():
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]
    return ti.Matrix(
        [
            [
                qw * qw + qx * qx - qy * qy - qz * qz,
                2 * (qx * qy - qw * qz),
                2 * (qx * qz + qw * qy),
            ],
            [
                2 * (qx * qy + qw * qz),
                (qw * qw - qx * qx + qy * qy - qz * qz),
                2 * (qy * qz - qw * qx),
            ],
            [
                2 * (qx * qz - qw * qy),
                2 * (qy * qz + qw * qx),
                qw * qw - qx * qx - qy * qy + qz * qz,
            ],
        ]
    )


@ti.kernel
def substep():
    # Compute the force on each particle
    pressure = 0
    for i in ti.grouped(particle_vertices):
        # gravity
        particle_force[i] = particle_mass * gravity

        is_in_contact = False
        # Collision force, we use a spring model to simulate the collision
        if particle_vertices[i][1] < -1:
            is_in_contact = True
            pressure = ti.cast(-particle_force[i][1], ti.i32)
            f_collision = collision_stiffness * (-1 - particle_vertices[i][1])
            particle_force[i] += ti.Vector([0, f_collision, 0])
        if particle_vertices[i][0] < -1:
            is_in_contact = True
            pressure = ti.cast(abs(particle_force[i][0]), ti.i32)
            f_collision = collision_stiffness * (-1 - particle_vertices[i][0])
            particle_force[i] += ti.Vector([f_collision, 0, 0])
        if particle_vertices[i][0] > 1:
            is_in_contact = True
            pressure = ti.cast(abs(particle_force[i][0]), ti.i32)
            f_collision = collision_stiffness * (1 - particle_vertices[i][0])
            particle_force[i] += ti.Vector([f_collision, 0, 0])
        if particle_vertices[i][2] < -1:
            is_in_contact = True
            pressure = ti.cast(abs(particle_force[i][2]), ti.i32)
            f_collision = collision_stiffness * (-1 + particle_vertices[i][2])
            particle_force[i] += ti.Vector([0, 0, f_collision])
        if particle_vertices[i][2] > 1:
            is_in_contact = True
            pressure = ti.cast(abs(particle_force[i][2]), ti.i32)
            f_collision = collision_stiffness * (1 - particle_vertices[i][2])
            particle_force[i] += ti.Vector([0, 0, f_collision])

        if is_in_contact:
            velocity_direction = particle_velocities[i] / particle_velocities[i].norm()
            friction_force = -friction_stiffness * pressure * velocity_direction
            particle_force[i] += friction_force

    # computer the force for rigid body
    body_force = ti.Vector([0.0, 0.0, 0.0])
    for i in ti.grouped(particle_vertices):
        # compute the force for rigid body
        body_force += particle_force[i]

    # computer the torque for rigid body
    body_torque = ti.Vector([0.0, 0.0, 0.0])
    for i in ti.grouped(particle_vertices):
        # compute the torque for rigid body
        # Hint: use ti.math.cross(v1, v2) to compute the cross product
        # torque +=
        r = particle_vertices[i] - body_cm_position[None]
        body_torque += ti.math.cross(r, particle_force[i])

    # update the rigid body
    # update the center of mass position and velocity
    body_velocity[None] += dt * body_force / body_mass[None]
    body_cm_position[None] += dt * body_velocity[None]

    # update the rotation quaternion
    d_q = 0.5 * quaternion_multiplication(
        ti.Vector(
            [
                0,
                body_angular_velocity[None][0],
                body_angular_velocity[None][1],
                body_angular_velocity[None][2],
            ]
        ),
        body_rotation_quaternion[None],
    )
    body_rotation_quaternion[None] += dt * d_q

    # normalize the quaternion to avoid numerical error
    body_rotation_quaternion[None] /= body_rotation_quaternion[None].norm()
    body_rotation[None] = quaternion_to_matrix(body_rotation_quaternion[None])

    # update, the angular momentum, inertia tensor and angular velocity
    body_angular_momentum[None] += dt * body_torque
    body_inverse_inertia = body_origin_inverse_inertia[None].transpose()
    body_angular_velocity[None] = body_inverse_inertia @ body_angular_momentum[None]

    # update the particles
    for i in ti.grouped(particle_vertices):
        ri = body_rotation[None] @ (
            particle_origin_vertices[i] - body_origin_cm_position[None]
        )
        particle_vertices[i] = ri + body_cm_position[None]
        particle_velocities[i] = body_velocity[None] + ti.math.cross(
            body_angular_velocity[None], ri
        )


# GUI stuff
# draw a cube frame
frame_vertices = ti.Vector.field(3, dtype=float, shape=24)
vertices_list = [
    [-1, -1, 0],
    [1, -1, 0],
    [1, -1, 0],
    [1, 1, 0],
    [1, 1, 0],
    [-1, 1, 0],
    [-1, 1, 0],
    [-1, -1, 0],
    [-1, -1, 1],
    [1, -1, 1],
    [1, -1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [-1, 1, 1],
    [-1, 1, 1],
    [-1, -1, 1],
    [-1, -1, 0],
    [-1, -1, 1],
    [1, -1, 0],
    [1, -1, 1],
    [1, 1, 0],
    [1, 1, 1],
    [-1, 1, 0],
    [-1, 1, 1],
]
for i in range(len(vertices_list)):
    frame_vertices[i] = ti.Vector(vertices_list[i])

window = ti.ui.Window("Rigid Body Simulation", (1024, 1024), fps_limit=60, vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
# rendering frame rate is 1/60
substeps = int(1 / 60 // dt)
# print(substeps)
current_t = 0.0

while window.running:
    for i in range(substeps):
        substep()
        current_t += dt

    camera.position(-3.0, 0.0, 2)
    camera.lookat(1.0, 0.0, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))

    scene.mesh(particle_vertices, indices=indices, two_sided=True, show_wireframe=True)

    scene.lines(frame_vertices, color=(1, 0, 0), width=1)
    canvas.scene(scene)
    window.show()
