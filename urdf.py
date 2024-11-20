from urdf2mjcf import run

run(
    urdf_path="goal.urdf",
    mjcf_path="goal.mjcf",
    copy_meshes=True,
)
