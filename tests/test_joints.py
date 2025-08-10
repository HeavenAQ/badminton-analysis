import pytest
from Joints import JOINTS, SKELETON_CONNECTIONS
from Types import COCOKeypoints


class TestJoints:
    def test_joints_structure(self):
        assert isinstance(JOINTS, dict)
        assert len(JOINTS) > 0

    def test_joints_contain_required_angles(self):
        required_joints = [
            "Left Elbow", "Right Elbow", "Left Knee", "Right Knee",
            "Left Shoulder", "Right Shoulder", "Left Crotch", "Right Crotch"
        ]
        for joint in required_joints:
            assert joint in JOINTS

    def test_joint_definition_structure(self):
        for joint_name, joint_def in JOINTS.items():
            assert isinstance(joint_def, tuple)
            assert len(joint_def) == 3
            for keypoint in joint_def:
                assert isinstance(keypoint, COCOKeypoints)

    def test_specific_joint_definitions(self):
        assert JOINTS["Left Elbow"] == (
            COCOKeypoints.LEFT_SHOULDER,
            COCOKeypoints.LEFT_ELBOW,
            COCOKeypoints.LEFT_WRIST,
        )
        assert JOINTS["Right Shoulder"] == (
            COCOKeypoints.RIGHT_HIP,
            COCOKeypoints.RIGHT_SHOULDER,
            COCOKeypoints.RIGHT_ELBOW,
        )


class TestSkeletonConnections:
    def test_skeleton_connections_structure(self):
        assert isinstance(SKELETON_CONNECTIONS, list)
        assert len(SKELETON_CONNECTIONS) > 0

    def test_skeleton_connection_format(self):
        for connection in SKELETON_CONNECTIONS:
            assert isinstance(connection, tuple)
            assert len(connection) == 2
            assert isinstance(connection[0], COCOKeypoints)
            assert isinstance(connection[1], COCOKeypoints)

    def test_skeleton_connections_contain_key_connections(self):
        shoulder_connection = (COCOKeypoints.LEFT_SHOULDER, COCOKeypoints.RIGHT_SHOULDER)
        hip_connection = (COCOKeypoints.LEFT_HIP, COCOKeypoints.RIGHT_HIP)
        
        assert shoulder_connection in SKELETON_CONNECTIONS
        assert hip_connection in SKELETON_CONNECTIONS