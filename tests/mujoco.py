#!/usr/bin/env python
# demonstration of markers (visual-only geoms)

import numpy as np
from mujoco_py import load_model_from_xml, MjSim, MjViewer

from simmod.modification.mujoco import MujocoBodyModifier, MujocoLightModifier

MODEL_XML = """
<?xml version="1.0" encoding="utf-8"?>
<mujoco>
   <worldbody>
      <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>    
      <body name="test_box" pos="0 0 1">
         <geom type="box" size=".1 .2 .3" rgba="0 .9 0 1"/>
      </body>
   </worldbody>
</mujoco>
"""

model = load_model_from_xml(MODEL_XML)
sim = MjSim(model)
viewer = MjViewer(sim)
body_rand = MujocoBodyModifier(sim)
#light_mod = MujocoLightModifier(sim)

value = np.asarray([-0.3, 0, 0])
val = 1
for i in range(1000):
    sim.step()
    ctrl = 0

    viewer.render()
    if i % 200 == 0 and i != 0:
        #body_rand.set_pos("test_box", value)
        #value *= -1

        if val == 1:
            val = 0.6
        else:
            val = 1

        model.light_active[-1] = val
        #model.body_pos[-1, 1] = goal
        #goal *= -1
        #sim.set_constants()
