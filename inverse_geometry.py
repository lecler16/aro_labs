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

import time
from pinocchio.utils import rotate


from tools import setcubeplacement

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)
    
    # Get target poses for both hands using the cube's hook frames
    left_target_pose = getcubeplacement(cube, LEFT_HOOK)   # Left hand target from cube's LEFT_HOOK frame
    right_target_pose = getcubeplacement(cube, RIGHT_HOOK) # Right hand target from cube's RIGHT_HOOK frame
    
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

    
def solve_dual_ik_3d(robot, left_target_pose, right_target_pose, left_frame_id, right_frame_id, q_init, max_iterations=200, tolerance=1e-2, viz=None):
    '''
    Solve dual-hand 3D inverse kinematics using iterative method
    Both hands move simultaneously to grasp the cube from opposite sides
    '''
    
    q = q_init.copy()
    best_q = q_init.copy()
    best_error_norm = float('inf')
    
    # print(f"Starting dual-hand IK")
    # print(f"Left target: {left_target_pose.translation}")
    # print(f"Right target: {right_target_pose.translation}")
    
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
        # if i % 50 == 0:
        # if right_error_norm < 0.1 and i%2 ==0:
        #     print(f"Iteration {i}: Total error = {error_norm:.4f}")
        #     print(f"  Left hand error: {left_error_norm:.4f}")
        #     print(f"  Right hand error: {right_error_norm:.4f}")
        
        # Update best solution found so far
        if error_norm < best_error_norm:
            best_error_norm = error_norm
            best_q = q.copy()

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
        if viz is not None and i % 1 == 0:
            viz.display(q)
            time.sleep(0.02)
    
    print(f"Dual-hand IK failed to converge after {max_iterations} iterations")
    # print(f"Final error: {error_norm:.4f}")
    
    # Return best solution found so far if it's reasonable
    print(f"Best error norm achieved: {best_error_norm:.4f}")
    return best_q #if best_error_norm < tolerance * 5 else None

    
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat()

    q = robot.q0.copy()
    
    # q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    # updatevisuals(viz, robot, cube, q0)

    # qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)    
    # updatevisuals(viz, robot, cube, qe)

    cube_placement_test = pin.SE3(rotate('z', 0.),np.array([0.33, 1.5, 0.93]))

    qtest,successinit = computeqgrasppose(robot, q, cube, cube_placement_test, viz)
    updatevisuals(viz, robot, cube, qtest)

