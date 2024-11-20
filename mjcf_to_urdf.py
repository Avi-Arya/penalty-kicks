import xml.etree.ElementTree as ET

def convert_mujoco_to_urdf(mujoco_path, urdf_path):
    def parse_body(body, parent_name="torso"):
        body_name = body.get("name", "unnamed")
        body_pos = body.get("pos", "0 0 0").split()

        # Add a link for this body
        link = ET.SubElement(robot, "link", name=body_name)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz=" ".join(body_pos), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value="1.0")
        ET.SubElement(inertial, "inertia", ixx="0.1", ixy="0", ixz="0", iyy="0.1", iyz="0", izz="0.1")

        # Process geoms as visuals and collisions
        for geom in body.findall("geom"):
            geom_type = geom.get("type", "box")
            geom_size = geom.get("size", "0.1 0.1 0.1").split()
            geom_fromto = geom.get("fromto", None)

            visual = ET.SubElement(link, "visual")
            collision = ET.SubElement(link, "collision")
            geometry_v = ET.SubElement(visual, "geometry")
            geometry_c = ET.SubElement(collision, "geometry")

            if geom_fromto:
                fromto_values = list(map(float, geom_fromto.split()))
                length = ((fromto_values[3] - fromto_values[0]) ** 2 +
                        (fromto_values[4] - fromto_values[1]) ** 2 +
                        (fromto_values[5] - fromto_values[2]) ** 2) ** 0.5
                ET.SubElement(geometry_v, "cylinder", radius=geom_size[0], length=str(length))
                ET.SubElement(geometry_c, "cylinder", radius=geom_size[0], length=str(length))
            else:
                if geom_type == "sphere":
                    ET.SubElement(geometry_v, "sphere", radius=geom_size[0])
                    ET.SubElement(geometry_c, "sphere", radius=geom_size[0])
                elif geom_type == "box":
                    ET.SubElement(geometry_v, "box", size=" ".join(geom_size))
                    ET.SubElement(geometry_c, "box", size=" ".join(geom_size))
                elif geom_type == "capsule":
                    ET.SubElement(geometry_v, "cylinder", radius=geom_size[0], length=geom_size[1])
                    ET.SubElement(geometry_c, "cylinder", radius=geom_size[0], length=geom_size[1])

        # Create a joint for this body, unless it's the root
        if parent_name != "torso" or body_name != "torso":
            joint_name = f"{body_name}_joint"
            joint = ET.SubElement(robot, "joint", name=joint_name, type="fixed")
            ET.SubElement(joint, "parent", link=parent_name)
            ET.SubElement(joint, "child", link=body_name)
            ET.SubElement(joint, "origin", xyz=" ".join(body_pos), rpy="0 0 0")

        # Recursively parse child bodies
        for child_body in body.findall("body"):
            parse_body(child_body, body_name)
    tree = ET.parse(mujoco_path)
    root = tree.getroot()

    # Create the root robot element
    robot = ET.Element("robot", name="humanoid_converted")

    # Add the torso as the main root link
    torso_link = ET.SubElement(robot, "link", name="torso")
    inertial = ET.SubElement(torso_link, "inertial")
    ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(inertial, "mass", value="5.0")
    ET.SubElement(inertial, "inertia", ixx="0.2", ixy="0", ixz="0", iyy="0.2", iyz="0", izz="0.2")

    # Parse all top-level bodies and attach to the torso
    for body in root.find("worldbody").findall("body"):
        parse_body(body, "torso")

    # Write the URDF to a file
    urdf_tree = ET.ElementTree(robot)
    urdf_tree.write(urdf_path, encoding="utf-8", xml_declaration=True)


# Convert the provided humanoid.xml to a URDF
mujoco_file = "humanoid.xml"  # Replace with the actual path
urdf_output_file = "humanoid3.urdf"
convert_mujoco_to_urdf(mujoco_file, urdf_output_file)
