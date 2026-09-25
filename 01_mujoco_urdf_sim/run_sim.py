import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("black_description.xml")
data = mujoco.MjData(model)

data.ctrl[:] = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()

