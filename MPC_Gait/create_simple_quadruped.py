

def create_simple_quadruped_xml():
    """Create a simple quadruped robot XML"""
    
    xml = """
    <mujoco model="simple_quadruped">
        <compiler angle="radian" coordinate="local"/>
        
        <option timestep="0.001" gravity="0 0 -9.81"/>
        
        <default>
            <geom rgba="0.8 0.6 0.4 1" friction="1 0.005 0.0001"/>
            <joint damping="5.0" armature="0.1"/>  </default>
        
        <asset>
            <texture name="grid" type="2d" builtin="checker" width="512" height="512"
                        rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
            <material name="grid" texture="grid" texrepeat="1 1" texuniform="true"
                        reflectance=".2"/>
        </asset>
        
        <worldbody>
            <light pos="0 0 1.5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
            <geom name="floor" pos="0 0 -0.1" size="5 5 .05" type="plane" material="grid"/>
            
            <body name="base" pos="0 0 0.5">
                <freejoint/>
                <geom name="torso" type="box" size="0.3 0.15 0.08" mass="18.0" rgba="0.2 0.2 0.8 1"/>
                <geom name="torso_imu" type="sphere" size="0.02" pos="0 0 0" mass="0.1" rgba="1 0 0 1"/>
                
                <body name="fl_hip" pos="0.27 0.17 -0.08">
                    <joint name="fl_hip_joint" type="hinge" axis="0 0 1" range="-0.5 0.5" damping="1.0"/>
                    <geom type="capsule" fromto="0 0 0 0 0.05 0" size="0.03" mass="1.0" rgba="0.8 0.2 0.2 1"/>
                    
                    <body name="fl_thigh" pos="0 0.05 0">
                        <joint name="fl_thigh_joint" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="1.0"/>
                        <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.025" mass="1.5" rgba="0.2 0.8 0.2 1"/>
                        
                        <body name="fl_shank" pos="0 0 -0.22">
                            <joint name="fl_shank_joint" type="hinge" axis="0 1 0" range="-2.5 -0.2" damping="1.0"/>
                            <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.02" mass="1.0" rgba="0.2 0.8 0.2 1"/>
                            <geom name="fl_foot" type="sphere" pos="0 0 -0.22" size="0.03" rgba="0.1 0.1 0.1 1"/>
                        </body>
                    </body>
                </body>
                
                <body name="fr_hip" pos="0.27 -0.17 -0.08">
                    <joint name="fr_hip_joint" type="hinge" axis="0 0 1" range="-0.5 0.5" damping="1.0"/>
                    <geom type="capsule" fromto="0 0 0 0 -0.05 0" size="0.03" mass="1.0" rgba="0.8 0.2 0.2 1"/>
                    
                    <body name="fr_thigh" pos="0 -0.05 0">
                        <joint name="fr_thigh_joint" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="1.0"/>
                        <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.025" mass="1.5" rgba="0.2 0.8 0.2 1"/>
                        
                        <body name="fr_shank" pos="0 0 -0.22">
                            <joint name="fr_shank_joint" type="hinge" axis="0 1 0" range="-2.5 -0.2" damping="1.0"/>
                            <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.02" mass="1.0" rgba="0.2 0.8 0.2 1"/>
                            <geom name="fr_foot" type="sphere" pos="0 0 -0.22" size="0.03" rgba="0.1 0.1 0.1 1"/>
                        </body>
                    </body>
                </body>
                
                <body name="hl_hip" pos="-0.27 0.17 -0.08">
                    <joint name="hl_hip_joint" type="hinge" axis="0 0 1" range="-0.5 0.5" damping="1.0"/>
                    <geom type="capsule" fromto="0 0 0 0 0.05 0" size="0.03" mass="1.0" rgba="0.8 0.2 0.2 1"/>
                    
                    <body name="hl_thigh" pos="0 0.05 0">
                        <joint name="hl_thigh_joint" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="1.0"/>
                        <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.025" mass="1.5" rgba="0.2 0.8 0.2 1"/>
                        
                        <body name="hl_shank" pos="0 0 -0.22">
                            <joint name="hl_shank_joint" type="hinge" axis="0 1 0" range="-2.5 -0.2" damping="1.0"/>
                            <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.02" mass="1.0" rgba="0.2 0.8 0.2 1"/>
                            <geom name="hl_foot" type="sphere" pos="0 0 -0.22" size="0.03" rgba="0.1 0.1 0.1 1"/>
                        </body>
                    </body>
                </body>
                
                <body name="hr_hip" pos="-0.27 -0.17 -0.08">
                    <joint name="hr_hip_joint" type="hinge" axis="0 0 1" range="-0.5 0.5" damping="1.0"/>
                    <geom type="capsule" fromto="0 0 0 0 -0.05 0" size="0.03" mass="1.0" rgba="0.8 0.2 0.2 1"/>
                    
                    <body name="hr_thigh" pos="0 -0.05 0">
                        <joint name="hr_thigh_joint" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="1.0"/>
                        <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.025" mass="1.5" rgba="0.2 0.8 0.2 1"/>
                        
                        <body name="hr_shank" pos="0 0 -0.22">
                            <joint name="hr_shank_joint" type="hinge" axis="0 1 0" range="-2.5 -0.2" damping="1.0"/>
                            <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.02" mass="1.0" rgba="0.2 0.8 0.2 1"/>
                            <geom name="hr_foot" type="sphere" pos="0 0 -0.22" size="0.03" rgba="0.1 0.1 0.1 1"/>
                        </body>
                    </body>
                </body>
            </body>
        </worldbody>
        
        <actuator>
            <motor name="fl_hip_motor" joint="fl_hip_joint" gear="50" ctrllimited="true" ctrlrange="-50 50"/>
            <motor name="fl_thigh_motor" joint="fl_thigh_joint" gear="150" ctrllimited="true" ctrlrange="-150 150"/>
            <motor name="fl_shank_motor" joint="fl_shank_joint" gear="150" ctrllimited="true" ctrlrange="-150 150"/>
            
            <motor name="fr_hip_motor" joint="fr_hip_joint" gear="50" ctrllimited="true" ctrlrange="-50 50"/>
            <motor name="fr_thigh_motor" joint="fr_thigh_joint" gear="150" ctrllimited="true" ctrlrange="-150 150"/>
            <motor name="fr_shank_motor" joint="fr_shank_joint" gear="150" ctrllimited="true" ctrlrange="-150 150"/>
            
            <motor name="hl_hip_motor" joint="hl_hip_joint" gear="50" ctrllimited="true" ctrlrange="-50 50"/>
            <motor name="hl_thigh_motor" joint="hl_thigh_joint" gear="150" ctrllimited="true" ctrlrange="-150 150"/>
            <motor name="hl_shank_motor" joint="hl_shank_joint" gear="150" ctrllimited="true" ctrlrange="-150 150"/>
            
            <motor name="hr_hip_motor" joint="hr_hip_joint" gear="50" ctrllimited="true" ctrlrange="-50 50"/>
            <motor name="hr_thigh_motor" joint="hr_thigh_joint" gear="150" ctrllimited="true" ctrlrange="-150 150"/>
            <motor name="hr_shank_motor" joint="hr_shank_joint" gear="150" ctrllimited="true" ctrlrange="-150 150"/>
        </actuator>
    </mujoco>
    """
    return xml

def create_simple_quadruped_xml_wheels():
    xml_with_wheels = """
        <mujoco model="simple_quadruped_wheels">
            <compiler angle="radian" coordinate="local"/>
            <option timestep="0.001" gravity="0 0 -9.81"/>

            <default>
                <geom rgba="0.8 0.6 0.4 1" friction="1 0.005 0.0001"/>
                <joint damping="5.0" armature="0.1"/>
            </default>

            <asset>
                <texture name="grid" type="2d" builtin="checker" width="512" height="512"
                        rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
                <material name="grid" texture="grid" texrepeat="1 1" texuniform="true"
                        reflectance=".2"/>
            </asset>

            <worldbody>
                <light pos="0 0 1.5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
                <geom name="floor" pos="0 0 -0.1" size="5 5 .05" type="plane" material="grid"/>

                <body name="base" pos="0 0 0.5">
                    <freejoint/>
                    <geom name="torso" type="box" size="0.3 0.15 0.08" mass="18.0" rgba="0.2 0.2 0.8 1"/>
                    <geom name="torso_imu" type="sphere" size="0.02" pos="0 0 0" mass="0.1" rgba="1 0 0 1"/>

                    <!-- ===== FRONT LEFT ===== -->
                    <body name="fl_hip" pos="0.27 0.17 -0.08">
                        <joint name="fl_hip_joint" type="hinge" axis="0 0 1" range="-0.5 0.5" damping="1.0"/>
                        <geom type="capsule" fromto="0 0 0 0 0.05 0" size="0.03" mass="1.0" rgba="0.8 0.2 0.2 1"/>

                        <body name="fl_thigh" pos="0 0.05 0">
                            <joint name="fl_thigh_joint" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="1.0"/>
                            <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.025" mass="1.5" rgba="0.2 0.8 0.2 1"/>

                            <body name="fl_shank" pos="0 0 -0.22">
                                <joint name="fl_shank_joint" type="hinge" axis="0 1 0" range="-2.5 -0.2" damping="1.0"/>
                                <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.02" mass="1.0" rgba="0.2 0.8 0.2 1"/>

                                <!-- wheel -->
                                <body name="fl_wheel" pos="0 0 -0.22">
                                    <joint name="fl_wheel_joint" type="hinge" axis="0 1 0" damping="0.01" range="-360 360"/>
                                    <geom name="fl_wheel_geom" type="cylinder" size="0.03 0.01"
                                        euler="1.5708 0 0" rgba="0.1 0.1 0.1 1"
                                        friction="1.2 0.003 0.00001"/>
                                </body>
                            </body>
                        </body>
                    </body>

                    <!-- ===== FRONT RIGHT ===== -->
                    <body name="fr_hip" pos="0.27 -0.17 -0.08">
                        <joint name="fr_hip_joint" type="hinge" axis="0 0 1" range="-0.5 0.5" damping="1.0"/>
                        <geom type="capsule" fromto="0 0 0 0 -0.05 0" size="0.03" mass="1.0" rgba="0.8 0.2 0.2 1"/>

                        <body name="fr_thigh" pos="0 -0.05 0">
                            <joint name="fr_thigh_joint" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="1.0"/>
                            <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.025" mass="1.5" rgba="0.2 0.8 0.2 1"/>

                            <body name="fr_shank" pos="0 0 -0.22">
                                <joint name="fr_shank_joint" type="hinge" axis="0 1 0" range="-2.5 -0.2" damping="1.0"/>
                                <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.02" mass="1.0" rgba="0.2 0.8 0.2 1"/>

                                <!-- wheel -->
                                <body name="fr_wheel" pos="0 0 -0.22">
                                    <joint name="fr_wheel_joint" type="hinge" axis="0 1 0" damping="0.01" range="-360 360"/>
                                    <geom name="fr_wheel_geom" type="cylinder" size="0.03 0.01"
                                        euler="1.5708 0 0" rgba="0.1 0.1 0.1 1"
                                        friction="1.2 0.003 0.00001"/>
                                </body>
                            </body>
                        </body>
                    </body>

                    <!-- ===== HIND LEFT ===== -->
                    <body name="hl_hip" pos="-0.27 0.17 -0.08">
                        <joint name="hl_hip_joint" type="hinge" axis="0 0 1" range="-0.5 0.5" damping="1.0"/>
                        <geom type="capsule" fromto="0 0 0 0 0.05 0" size="0.03" mass="1.0" rgba="0.8 0.2 0.2 1"/>

                        <body name="hl_thigh" pos="0 0.05 0">
                            <joint name="hl_thigh_joint" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="1.0"/>
                            <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.025" mass="1.5" rgba="0.2 0.8 0.2 1"/>

                            <body name="hl_shank" pos="0 0 -0.22">
                                <joint name="hl_shank_joint" type="hinge" axis="0 1 0" range="-2.5 -0.2" damping="1.0"/>
                                <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.02" mass="1.0" rgba="0.2 0.8 0.2 1"/>

                                <!-- wheel -->
                                <body name="hl_wheel" pos="0 0 -0.22">
                                    <joint name="hl_wheel_joint" type="hinge" axis="0 1 0" damping="0.01" range="-360 360"/>
                                    <geom name="hl_wheel_geom" type="cylinder" size="0.03 0.01"
                                        euler="1.5708 0 0" rgba="0.1 0.1 0.1 1"
                                        friction="1.2 0.003 0.00001"/>
                                </body>
                            </body>
                        </body>
                    </body>

                    <!-- ===== HIND RIGHT ===== -->
                    <body name="hr_hip" pos="-0.27 -0.17 -0.08">
                        <joint name="hr_hip_joint" type="hinge" axis="0 0 1" range="-0.5 0.5" damping="1.0"/>
                        <geom type="capsule" fromto="0 0 0 0 -0.05 0" size="0.03" mass="1.0" rgba="0.8 0.2 0.2 1"/>

                        <body name="hr_thigh" pos="0 -0.05 0">
                            <joint name="hr_thigh_joint" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="1.0"/>
                            <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.025" mass="1.5" rgba="0.2 0.8 0.2 1"/>

                            <body name="hr_shank" pos="0 0 -0.22">
                                <joint name="hr_shank_joint" type="hinge" axis="0 1 0" range="-2.5 -0.2" damping="1.0"/>
                                <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.02" mass="1.0" rgba="0.2 0.8 0.2 1"/>

                                <!-- wheel -->
                                <body name="hr_wheel" pos="0 0 -0.22">
                                    <joint name="hr_wheel_joint" type="hinge" axis="0 1 0" damping="0.01" range="-360 360"/>
                                    <geom name="hr_wheel_geom" type="cylinder" size="0.03 0.01"
                                        euler="1.5708 0 0" rgba="0.1 0.1 0.1 1"
                                        friction="1.2 0.003 0.00001"/>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </worldbody>

            <actuator>
                <!-- leg motors -->
                <motor name="fl_hip_motor" joint="fl_hip_joint" gear="50" ctrllimited="true" ctrlrange="-50 50"/>
                <motor name="fl_thigh_motor" joint="fl_thigh_joint" gear="150" ctrllimited="true" ctrlrange="-150 150"/>
                <motor name="fl_shank_motor" joint="fl_shank_joint" gear="150" ctrllimited="true" ctrlrange="-150 150"/>

                <motor name="fr_hip_motor" joint="fr_hip_joint" gear="50" ctrllimited="true" ctrlrange="-50 50"/>
                <motor name="fr_thigh_motor" joint="fr_thigh_joint" gear="150" ctrllimited="true" ctrlrange="-150 150"/>
                <motor name="fr_shank_motor" joint="fr_shank_joint" gear="150" ctrllimited="true" ctrlrange="-150 150"/>

                <motor name="hl_hip_motor" joint="hl_hip_joint" gear="50" ctrllimited="true" ctrlrange="-50 50"/>
                <motor name="hl_thigh_motor" joint="hl_thigh_joint" gear="150" ctrllimited="true" ctrlrange="-150 150"/>
                <motor name="hl_shank_motor" joint="hl_shank_joint" gear="150" ctrllimited="true" ctrlrange="-150 150"/>

                <motor name="hr_hip_motor" joint="hr_hip_joint" gear="50" ctrllimited="true" ctrlrange="-50 50"/>
                <motor name="hr_thigh_motor" joint="hr_thigh_joint" gear="150" ctrllimited="true" ctrlrange="-150 150"/>
                <motor name="hr_shank_motor" joint="hr_shank_joint" gear="150" ctrllimited="true" ctrlrange="-150 150"/>

                <!-- optional wheel actuators (set ctrl=0 for passive rolling) -->
                <motor name="fl_wheel_motor" joint="fl_wheel_joint" gear="10" ctrllimited="true" ctrlrange="-5 5"/>
                <motor name="fr_wheel_motor" joint="fr_wheel_joint" gear="10" ctrllimited="true" ctrlrange="-5 5"/>
                <motor name="hl_wheel_motor" joint="hl_wheel_joint" gear="10" ctrllimited="true" ctrlrange="-5 5"/>
                <motor name="hr_wheel_motor" joint="hr_wheel_joint" gear="10" ctrllimited="true" ctrlrange="-5 5"/>
            </actuator>
        </mujoco>

    """
    return xml_with_wheels
