#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import pinocchio as pin 
import numpy as np
from numpy.linalg import pinv,inv,norm,svd,eig
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

from tools import setcubeplacement

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)
    
    # Extract target pose (position + orientation) from SE3 transformation
    target_position = cubetarget.translation  # [x, y, z]
    target_rotation = cubetarget.rotation     # 3x3 rotation matrix
    
    print(f"Target position: {target_position}")
    print(f"Target rotation matrix:\n{target_rotation}")

    # Define grasp poses for both hands relative to the cube
    # Left hand approaches from the left side, right hand from the right side
    cube_size = 0.05  # Approximate cube size
    grasp_offset = 0.08  # Distance from cube center to hand
    
    # Create target poses for both hands
    left_target_pos = target_position + np.array([-grasp_offset, 0, 0])  # Left side
    right_target_pos = target_position + np.array([grasp_offset, 0, 0])  # Right side
    
    # Both hands approach with same orientation as cube
    left_target_pose = pin.SE3(target_rotation, left_target_pos)
    right_target_pose = pin.SE3(target_rotation, right_target_pos)
    
    print(f"Left hand target: {left_target_pos}")
    print(f"Right hand target: {right_target_pos}")
    
    # Get frame IDs for both hands
    try:
        left_frame_id = robot.model.getFrameId(LEFT_HAND)
        right_frame_id = robot.model.getFrameId(RIGHT_HAND)
    except RuntimeError as e:
        print(f"Frame not found: {e}")
        return robot.q0, False
    
    # Solve dual-hand inverse kinematics
    q_solution = solve_dual_ik_3d(robot, left_target_pose, right_target_pose, 
                                  left_frame_id, right_frame_id, qcurrent, viz=viz)
    
    if q_solution is not None:
        return q_solution, True

    return robot.q0, False
    
def solve_ik_3d(robot, target_pose, frame_id, q_init, max_iterations=500, tolerance=1e-2, viz=None):
    '''
    Solve 3D inverse kinematics using improved iterative method
    Based on decoupled approach: solve orientation first, then position
    '''
    
    q = q_init.copy()
    
    print(f"Starting IK with target position: {target_pose.translation}")
    
    # First, try a more robust approach with better step control
    for i in range(max_iterations):
        # Update robot kinematics
        pin.framesForwardKinematics(robot.model, robot.data, q)
        pin.computeJointJacobians(robot.model, robot.data, q)
        
        # Get current end effector pose
        current_pose = robot.data.oMf[frame_id]
        
        # Compute error (from current to target) - CORRECTED
        position_error = target_pose.translation - current_pose.translation
        rotation_error = pin.log3(current_pose.rotation.T @ target_pose.rotation)
        
        # Combine errors
        error_6d = np.concatenate([position_error, rotation_error])
        error_norm = np.linalg.norm(error_6d)
        position_error_norm = np.linalg.norm(position_error)
        rotation_error_norm = np.linalg.norm(rotation_error)
        
        # Print progress
        if i % 50 == 0 or i < 10:
            print(f"Iteration {i}: Total error = {error_norm:.4f}")
            print(f"  Position error: {position_error_norm:.4f}")
            print(f"  Rotation error: {rotation_error_norm:.4f}")
        
        # Check convergence
        if error_norm < tolerance:
            print(f"IK converged in {i+1} iterations with error {error_norm:.4f}")
            return q
        
        # Get Jacobian for the end effector frame
        jacobian = pin.getFrameJacobian(robot.model, robot.data, frame_id, pin.LOCAL_WORLD_ALIGNED)
        jacobian_6d = jacobian[:6, :]
        
        # Improved damped least squares with adaptive damping
        if error_norm > 0.1:
            damping = 1e-3  # Higher damping for large errors
            step_size = 0.3
        elif error_norm > 0.05:
            damping = 1e-4
            step_size = 0.5
        else:
            damping = 1e-5
            step_size = 0.8
        
        # Compute damped pseudo-inverse
        jacobian_T = jacobian_6d.T
        jacobian_pinv = jacobian_T @ np.linalg.inv(jacobian_6d @ jacobian_T + damping * np.eye(6))
        
        # Compute joint update
        dq = jacobian_pinv @ error_6d
        
        # Limit step size to prevent large jumps
        max_joint_step = 0.2  # Increased from 0.1
        dq_norm = np.linalg.norm(dq)
        if dq_norm > max_joint_step:
            dq = dq * (max_joint_step / dq_norm)
        
        # Apply update
        q_new = q + step_size * dq
        
        # Project to joint limits
        q_new = projecttojointlimits(robot, q_new)
        
        # Check if update improves the solution
        pin.framesForwardKinematics(robot.model, robot.data, q_new)
        new_pose = robot.data.oMf[frame_id]
        new_pos_error = np.linalg.norm(target_pose.translation - new_pose.translation)
        new_rot_error = np.linalg.norm(pin.log3(new_pose.rotation.T @ target_pose.rotation))
        new_total_error = new_pos_error + new_rot_error
        
        current_total_error = position_error_norm + rotation_error_norm
        
        # Accept update if it improves or doesn't worsen significantly
        if new_total_error <= current_total_error * 1.1:  # Allow 10% worsening
            q = q_new
        else:
            # Reduce step size if update doesn't improve
            step_size *= 0.5
            if step_size < 1e-6:
                print(f"Step size too small, stopping at iteration {i}")
                break
            q = q + step_size * dq
            q = projecttojointlimits(robot, q)
        
        # Visualize progress
        if viz is not None and i % 25 == 0:
            viz.display(q)
    
    print(f"IK failed to converge after {max_iterations} iterations")
    print(f"Final error: {error_norm:.4f}")
    
    # Return best solution found so far
    return q if error_norm < tolerance * 10 else None
    
def solve_dual_ik_3d(robot, left_target_pose, right_target_pose, left_frame_id, right_frame_id, q_init, max_iterations=500, tolerance=1e-2, viz=None):
    '''
    Solve dual-hand 3D inverse kinematics using iterative method
    Both hands move simultaneously to grasp the cube from opposite sides
    '''
    
    q = q_init.copy()
    
    print(f"Starting dual-hand IK")
    print(f"Left target: {left_target_pose.translation}")
    print(f"Right target: {right_target_pose.translation}")
    
    for i in range(max_iterations):
        # Update robot kinematics
        pin.framesForwardKinematics(robot.model, robot.data, q)
        pin.computeJointJacobians(robot.model, robot.data, q)
        
        # Get current poses for both hands
        left_current_pose = robot.data.oMf[left_frame_id]
        right_current_pose = robot.data.oMf[right_frame_id]
        
        # Compute errors for both hands
        left_pos_error = left_target_pose.translation - left_current_pose.translation
        left_rot_error = pin.log3(left_current_pose.rotation.T @ left_target_pose.rotation)
        
        right_pos_error = right_target_pose.translation - right_current_pose.translation
        right_rot_error = pin.log3(right_current_pose.rotation.T @ right_target_pose.rotation)
        
        # Combine all errors into 12D vector (6D for each hand)
        left_error_6d = np.concatenate([left_pos_error, left_rot_error])
        right_error_6d = np.concatenate([right_pos_error, right_rot_error])
        error_12d = np.concatenate([left_error_6d, right_error_6d])
        
        error_norm = np.linalg.norm(error_12d)
        left_error_norm = np.linalg.norm(left_error_6d)
        right_error_norm = np.linalg.norm(right_error_6d)
        
        # Print progress
        if i % 50 == 0 or i < 10:
            print(f"Iteration {i}: Total error = {error_norm:.4f}")
            print(f"  Left hand error: {left_error_norm:.4f}")
            print(f"  Right hand error: {right_error_norm:.4f}")
        
        # Check convergence
        if error_norm < tolerance:
            print(f"Dual-hand IK converged in {i+1} iterations with error {error_norm:.4f}")
            return q
        
        # Get Jacobians for both hands
        left_jacobian = pin.getFrameJacobian(robot.model, robot.data, left_frame_id, pin.LOCAL_WORLD_ALIGNED)
        right_jacobian = pin.getFrameJacobian(robot.model, robot.data, right_frame_id, pin.LOCAL_WORLD_ALIGNED)
        
        # Use only position and orientation (6D) for each hand
        left_jacobian_6d = left_jacobian[:6, :]
        right_jacobian_6d = right_jacobian[:6, :]
        
        # Stack Jacobians to create 12 x n_joints matrix
        jacobian_12d = np.vstack([left_jacobian_6d, right_jacobian_6d])
        
        # Adaptive damping based on error magnitude
        if error_norm > 0.1:
            damping = 1e-3
            step_size = 0.3
        elif error_norm > 0.05:
            damping = 1e-4
            step_size = 0.5
        else:
            damping = 1e-5
            step_size = 0.8
        
        # Compute damped pseudo-inverse
        jacobian_T = jacobian_12d.T
        jacobian_pinv = jacobian_T @ np.linalg.inv(jacobian_12d @ jacobian_T + damping * np.eye(12))
        
        # Compute joint update
        dq = jacobian_pinv @ error_12d
        
        # Limit step size to prevent large jumps
        max_joint_step = 0.2
        dq_norm = np.linalg.norm(dq)
        if dq_norm > max_joint_step:
            dq = dq * (max_joint_step / dq_norm)
        
        # Apply update
        q_new = q + step_size * dq
        
        # Project to joint limits
        q_new = projecttojointlimits(robot, q_new)
        
        # Check if update improves the solution
        pin.framesForwardKinematics(robot.model, robot.data, q_new)
        new_left_pose = robot.data.oMf[left_frame_id]
        new_right_pose = robot.data.oMf[right_frame_id]
        
        new_left_pos_error = np.linalg.norm(left_target_pose.translation - new_left_pose.translation)
        new_left_rot_error = np.linalg.norm(pin.log3(new_left_pose.rotation.T @ left_target_pose.rotation))
        new_right_pos_error = np.linalg.norm(right_target_pose.translation - new_right_pose.translation)
        new_right_rot_error = np.linalg.norm(pin.log3(new_right_pose.rotation.T @ right_target_pose.rotation))
        
        new_total_error = new_left_pos_error + new_left_rot_error + new_right_pos_error + new_right_rot_error
        current_total_error = left_error_norm + right_error_norm
        
        # Accept update if it improves or doesn't worsen significantly
        if new_total_error <= current_total_error * 1.1:  # Allow 10% worsening
            q = q_new
        else:
            # Reduce step size if update doesn't improve
            step_size *= 0.5
            if step_size < 1e-6:
                print(f"Step size too small, stopping at iteration {i}")
                break
            q = q + step_size * dq
            q = projecttojointlimits(robot, q)
        
        # Visualize progress
        if viz is not None and i % 25 == 0:
            viz.display(q)
    
    print(f"Dual-hand IK failed to converge after {max_iterations} iterations")
    print(f"Final error: {error_norm:.4f}")
    
    # Return best solution found so far if it's reasonable
    return q if error_norm < tolerance * 5 else None

    
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat()

    q = robot.q0.copy()
    
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    # qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    updatevisuals(viz, robot, cube, q0)

