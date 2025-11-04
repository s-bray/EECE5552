"""
Convert ANYmal URDF to MuJoCo XML with automatic fixes
"""

import os
import sys


def convert_anymal_urdf_to_mujoco():
    """
    Convert ANYmal URDF to MuJoCo XML format
    
    This uses MuJoCo's built-in URDF compiler with automatic fixes
    """
    
    print("="*60)
    print("ANYmal URDF to MuJoCo XML Converter")
    print("="*60)
    
    # Paths
    urdf_path = "/home/poison-arrow/MPC_Gait/anymal_with_wheels-main/aww_description/urdf/anymal_with_wheels.urdf"
    output_path = "/home/poison-arrow/MPC_Gait/anymal_mujoco.xml"
    
    if not os.path.exists(urdf_path):
        print(f"✗ URDF not found: {urdf_path}")
        print("\nPlease update the urdf_path variable in this script.")
        return False
    
    print(f"\n📄 Input URDF: {urdf_path}")
    print(f"📄 Output XML: {output_path}")
    
    try:
        import mujoco
        print(f"\n✓ MuJoCo version: {mujoco.__version__}")
        
        print("\n⚙ Converting URDF to MuJoCo XML...")
        print("  (This may show warnings - that's normal)")
        
        # Load URDF with automatic fixes
        # balanceinertia=True fixes inertia problems automatically
        model = mujoco.MjModel.from_xml_path(
            urdf_path,
            {'balanceinertia': 'true'}  # Automatically fix inertia issues
        )
        
        print("\n✓ URDF loaded successfully!")
        print(f"  • Bodies: {model.nbody}")
        print(f"  • Joints: {model.njnt}")
        print(f"  • Actuators: {model.nu}")
        
        # Save as MuJoCo XML
        print(f"\n💾 Saving to: {output_path}")
        mujoco.mj_saveLastXML(output_path, model)
        
        print("\n" + "="*60)
        print("✓ CONVERSION SUCCESSFUL!")
        print("="*60)
        print(f"\nYou can now use the converted model:")
        print(f"  ROBOT_XML_PATH = '{output_path}'")
        print(f"  USE_SIMPLE_ROBOT = False")
        
        return True
        
    except ImportError:
        print("\n✗ MuJoCo not installed")
        print("Install with: pip install mujoco")
        return False
        
    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check that mesh files exist in the URDF package")
        print("2. Verify URDF is well-formed")
        print("3. Try using the simple quadruped model instead")
        return False


def create_standalone_anymal_xml():
    """
    Create a simplified ANYmal-like quadruped from scratch
    This avoids URDF conversion issues entirely
    """
    
    print("\n" + "="*60)
    print("Creating Simplified ANYmal XML")
    print("="*60)
    
    xml_content = """
    <mujoco model="anymal_simplified">
        <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>
        
        <option timestep="0.001" gravity="0 0 -9.81" iterations="50" solver="Newton" tolerance="1e-10"/>
        
        <default>
            <geom rgba="0.7 0.7 0.7 1" friction="1.0 0.005 0.0001" condim="3"/>
            <joint damping="1.0" armature="0.01" frictionloss="0.1"/>
            <motor ctrlrange="-200 200" ctrllimited="true"/>
        </default>
        
        <asset>
            <texture name="grid" type="2d" builtin="checker" width="512" height="512"
                     rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
            <material name="grid" texture="grid" texrepeat="2 2" texuniform="true" reflectance=".2"/>
            <texture name="skybox" type="skybox" builtin="gradient" rgb1=".4 .6 .8" rgb2="0 0 0" 
                     width="800" height="800" mark="random" markrgb="1 1 1"/>
        </asset>
        
        <worldbody>
            <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>
            <geom name="floor" size="20 20 .05" type="plane" material="grid"/>
            
            <!-- ANYmal-like Base -->
            <body name="base" pos="0 0 0.55">
                <freejoint/>
                <inertial pos="0 0 0" mass="16.0" diaginertia="0.5 1.0 1.0"/>
                <geom name="torso" type="box" size="0.28 0.14 0.06" rgba="0.2 0.2 0.8 1"/>
                
                <!-- LF Leg (Left Front) -->
                <body name="LF_HIP" pos="0.277 0.116 0">
                    <inertial pos="0 0.04 0" mass="1.42" diaginertia="0.01 0.01 0.01"/>
                    <joint name="LF_HAA" type="hinge" axis="1 0 0" range="-0.4 0.4" damping="2.0"/>
                    <geom type="capsule" fromto="0 0 0 0 0.08 0" size="0.04" rgba="0.8 0.2 0.2 1"/>
                    
                    <body name="LF_THIGH" pos="0 0.1 0">
                        <inertial pos="0 0 -0.125" mass="1.634" diaginertia="0.02 0.02 0.001"/>
                        <joint name="LF_HFE" type="hinge" axis="0 1 0" range="-1.5 1.5" damping="2.0"/>
                        <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.035" rgba="0.2 0.8 0.2 1"/>
                        
                        <body name="LF_SHANK" pos="0 0 -0.25">
                            <inertial pos="0 0 -0.125" mass="0.207" diaginertia="0.005 0.005 0.0001"/>
                            <joint name="LF_KFE" type="hinge" axis="0 1 0" range="-2.6 -0.3" damping="1.0"/>
                            <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.025" rgba="0.2 0.8 0.2 1"/>
                            <geom name="LF_FOOT" type="sphere" pos="0 0 -0.25" size="0.035" rgba="0.1 0.1 0.1 1" friction="1.5 0.01 0.001"/>
                        </body>
                    </body>
                </body>
                
                <!-- RF Leg (Right Front) -->
                <body name="RF_HIP" pos="0.277 -0.116 0">
                    <inertial pos="0 -0.04 0" mass="1.42" diaginertia="0.01 0.01 0.01"/>
                    <joint name="RF_HAA" type="hinge" axis="1 0 0" range="-0.4 0.4" damping="2.0"/>
                    <geom type="capsule" fromto="0 0 0 0 -0.08 0" size="0.04" rgba="0.8 0.2 0.2 1"/>
                    
                    <body name="RF_THIGH" pos="0 -0.1 0">
                        <inertial pos="0 0 -0.125" mass="1.634" diaginertia="0.02 0.02 0.001"/>
                        <joint name="RF_HFE" type="hinge" axis="0 1 0" range="-1.5 1.5" damping="2.0"/>
                        <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.035" rgba="0.2 0.8 0.2 1"/>
                        
                        <body name="RF_SHANK" pos="0 0 -0.25">
                            <inertial pos="0 0 -0.125" mass="0.207" diaginertia="0.005 0.005 0.0001"/>
                            <joint name="RF_KFE" type="hinge" axis="0 1 0" range="-2.6 -0.3" damping="1.0"/>
                            <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.025" rgba="0.2 0.8 0.2 1"/>
                            <geom name="RF_FOOT" type="sphere" pos="0 0 -0.25" size="0.035" rgba="0.1 0.1 0.1 1" friction="1.5 0.01 0.001"/>
                        </body>
                    </body>
                </body>
                
                <!-- LH Leg (Left Hind) -->
                <body name="LH_HIP" pos="-0.277 0.116 0">
                    <inertial pos="0 0.04 0" mass="1.42" diaginertia="0.01 0.01 0.01"/>
                    <joint name="LH_HAA" type="hinge" axis="1 0 0" range="-0.4 0.4" damping="2.0"/>
                    <geom type="capsule" fromto="0 0 0 0 0.08 0" size="0.04" rgba="0.8 0.2 0.2 1"/>
                    
                    <body name="LH_THIGH" pos="0 0.1 0">
                        <inertial pos="0 0 -0.125" mass="1.634" diaginertia="0.02 0.02 0.001"/>
                        <joint name="LH_HFE" type="hinge" axis="0 1 0" range="-1.5 1.5" damping="2.0"/>
                        <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.035" rgba="0.2 0.8 0.2 1"/>
                        
                        <body name="LH_SHANK" pos="0 0 -0.25">
                            <inertial pos="0 0 -0.125" mass="0.207" diaginertia="0.005 0.005 0.0001"/>
                            <joint name="LH_KFE" type="hinge" axis="0 1 0" range="-2.6 -0.3" damping="1.0"/>
                            <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.025" rgba="0.2 0.8 0.2 1"/>
                            <geom name="LH_FOOT" type="sphere" pos="0 0 -0.25" size="0.035" rgba="0.1 0.1 0.1 1" friction="1.5 0.01 0.001"/>
                        </body>
                    </body>
                </body>
                
                <!-- RH Leg (Right Hind) -->
                <body name="RH_HIP" pos="-0.277 -0.116 0">
                    <inertial pos="0 -0.04 0" mass="1.42" diaginertia="0.01 0.01 0.01"/>
                    <joint name="RH_HAA" type="hinge" axis="1 0 0" range="-0.4 0.4" damping="2.0"/>
                    <geom type="capsule" fromto="0 0 0 0 -0.08 0" size="0.04" rgba="0.8 0.2 0.2 1"/>
                    
                    <body name="RH_THIGH" pos="0 -0.1 0">
                        <inertial pos="0 0 -0.125" mass="1.634" diaginertia="0.02 0.02 0.001"/>
                        <joint name="RH_HFE" type="hinge" axis="0 1 0" range="-1.5 1.5" damping="2.0"/>
                        <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.035" rgba="0.2 0.8 0.2 1"/>
                        
                        <body name="RH_SHANK" pos="0 0 -0.25">
                            <inertial pos="0 0 -0.125" mass="0.207" diaginertia="0.005 0.005 0.0001"/>
                            <joint name="RH_KFE" type="hinge" axis="0 1 0" range="-2.6 -0.3" damping="1.0"/>
                            <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.025" rgba="0.2 0.8 0.2 1"/>
                            <geom name="RH_FOOT" type="sphere" pos="0 0 -0.25" size="0.035" rgba="0.1 0.1 0.1 1" friction="1.5 0.01 0.001"/>
                        </body>
                    </body>
                </body>
            </body>
        </worldbody>
        
        <actuator>
            <!-- Hip Ab/Adduction -->
            <motor name="LF_HAA_motor" joint="LF_HAA" gear="50"/>
            <motor name="RF_HAA_motor" joint="RF_HAA" gear="50"/>
            <motor name="LH_HAA_motor" joint="LH_HAA" gear="50"/>
            <motor name="RH_HAA_motor" joint="RH_HAA" gear="50"/>
            
            <!-- Hip Flexion/Extension -->
            <motor name="LF_HFE_motor" joint="LF_HFE" gear="150"/>
            <motor name="RF_HFE_motor" joint="RF_HFE" gear="150"/>
            <motor name="LH_HFE_motor" joint="LH_HFE" gear="150"/>
            <motor name="RH_HFE_motor" joint="RH_HFE" gear="150"/>
            
            <!-- Knee Flexion/Extension -->
            <motor name="LF_KFE_motor" joint="LF_KFE" gear="150"/>
            <motor name="RF_KFE_motor" joint="RF_KFE" gear="150"/>
            <motor name="LH_KFE_motor" joint="LH_KFE" gear="150"/>
            <motor name="RH_KFE_motor" joint="RH_KFE" gear="150"/>
        </actuator>
        
        <sensor>
            <!-- IMU -->
            <gyro name="base_gyro" site="base"/>
            <accelerometer name="base_accel" site="base"/>
            
            <!-- Joint sensors -->
            <jointpos name="LF_HAA_pos" joint="LF_HAA"/>
            <jointvel name="LF_HAA_vel" joint="LF_HAA"/>
        </sensor>
    </mujoco>
    """
    
    output_path = "/home/poison-arrow/MPC_Gait/anymal_simplified.xml"
    
    try:
        with open(output_path, 'w') as f:
            f.write(xml_content)
        
        print(f"\n✓ Created: {output_path}")
        print("\nThis is a simplified ANYmal-like model that should work reliably.")
        print("It has:")
        print("  • Proper mass distribution (~30 kg total)")
        print("  • 12 actuated joints (3 per leg)")
        print("  • Realistic joint ranges")
        print("  • Good friction parameters")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Failed to create XML: {e}")
        return False


if __name__ == "__main__":
    print("\nChoose an option:")
    print("1. Convert existing ANYmal URDF to MuJoCo XML")
    print("2. Create simplified ANYmal XML from scratch (Recommended)")
    print("3. Both")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        convert_anymal_urdf_to_mujoco()
    elif choice == "2":
        create_standalone_anymal_xml()
    elif choice == "3":
        create_standalone_anymal_xml()
        print("\n" + "="*60)
        convert_anymal_urdf_to_mujoco()
    else:
        print("Invalid choice. Running option 2 (recommended)...")
        create_standalone_anymal_xml()