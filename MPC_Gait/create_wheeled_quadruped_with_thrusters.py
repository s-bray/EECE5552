"""
Create a wheeled quadruped robot model with thrusters.

This version has:
- Wheels at the end of legs (like the original)
- 4 thrusters at torso corners
- Standard 3-DOF legs (hip, thigh, shank)
- Perfect for testing thruster stabilization with wheels
"""
from create_simple_quadruped import create_simple_quadruped_xml_wheels


if __name__ == "__main__":
    # Test: generate and save the model
    xml_content = create_simple_quadruped_xml_wheels()
    
    output_file = "wheeled_quadruped_with_thrusters.xml"
    with open(output_file, 'w') as f:
        f.write(xml_content)
    
    print(f"✓ Created wheeled quadruped with thrusters: {output_file}")
    print("\nModel features:")
    print("  • 4 legs with 3 DOF each (hip, thigh, shank)")
    print("  • 4 wheels at leg ends (free-rolling)")
    print("  • 4 thrusters at torso corners")
    print("  • Red boxes mark thruster locations")
    print("  • Total mass: ~16 kg (with wheels)")
    print("  • Wheel friction: 0.8 (good rolling)")
    print("\nUse this model to test thruster stabilization with wheels!")
