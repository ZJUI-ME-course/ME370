import numpy as np
from scipy.optimize import fsolve

def solve_general_kinematics(R1, R2, P_c, V_c, A_c):
    """
    根据给定的驱动点C的完整2D运动状态，求解机构的运动学。

    参数:
    R1 (float): 连杆1的长度 (连接驱动点C)
    R2 (float): 连杆2的长度 (连接地面)
    P_c (tuple): C点的位置矢量 (xc, yc)
    V_c (tuple): C点的速度矢量 (v_cx, v_cy)
    A_c (tuple): C点的加速度矢量 (a_cx, a_cy)

    返回:
    一个包含所有求解结果的字典
    """
    xc, yc = P_c
    v_cx, v_cy = V_c
    a_cx, a_cy = A_c

    # --- 1. 位置分析 (求解 theta_1, theta_2) ---
    # 使用数值求解器求解非线性方程组
    # xc+R1*cos(th1) - R2*cos(th2) = 1.5
    # yc+R1*sin(th1) - R2*sin(th2) = 0c
    def position_equations(vars):
        th1, th2 = vars
        eq1 = R1 * np.cos(th1) - R2 * np.cos(th2) + xc-1.5
        eq2 = R1 * np.sin(th1) - R2 * np.sin(th2) + yc
        return [eq1, eq2]

    # 提供一个初始猜测值 (radians) 来寻找一个解
    # 根据 P_c 在第二象限，猜测 R1 指向右上方，R2 指向左上方
    initial_guess = np.deg2rad([45, 135]) 
    try:
        th1, th2 = fsolve(position_equations, initial_guess)
    except Exception as e:
        print(f"位置分析求解失败: {e}")
        return None

    s1, c1 = np.sin(th1), np.cos(th1)
    s2, c2 = np.sin(th2), np.cos(th2)

    # --- 2. 速度分析 (求解 omega_1, omega_2) ---
    # 建立线性方程组: A_mat * w = b_vel
    # [-R1*s1, R2*s2] [w1] = [-v_cx]
    # [ R1*c1,-R2*c2] [w2] = [-v_cy]
    A_mat = np.array([
        [-R1 * s1, R2 * s2],
        [ R1 * c1, -R2 * c2]
    ])
    b_vel = np.array([-v_cx, -v_cy])
    
    try:
        w1, w2 = np.linalg.solve(A_mat, b_vel)
    except np.linalg.LinAlgError:
        print("错误: 速度分析矩阵是奇异的，无法求解。机构可能处于死点位置。")
        return None

    # --- 3. 加速度分析 (求解 alpha_1, alpha_2) ---
    # 建立线性方程组: A_mat * alpha = b_acc
    # A_mat 矩阵与速度分析相同
    # b_acc_x = -a_cx + w1^2*R1*c1 - w2^2*R2*c2
    # b_acc_y = -a_cy + w1^2*R1*s1 - w2^2*R2*s2
    b_acc = np.array([
        -a_cx + w1**2 * R1 * c1 - w2**2 * R2 * c2,
        -a_cy + w1**2 * R1 * s1 - w2**2 * R2 * s2
    ])
    
    try:
        alpha1, alpha2 = np.linalg.solve(A_mat, b_acc)
    except np.linalg.LinAlgError:
        print("错误: 加速度分析矩阵是奇异的，无法求解。机构可能处于死点位置。")
        return None

    results = {
        'theta_1_deg': np.rad2deg(th1),
        'theta_2_deg': np.rad2deg(th2),
        'omega_1_rad_s': w1,
        'omega_2_rad_s': w2,
        'alpha_1_rad_s2': alpha1,
        'alpha_2_rad_s2': alpha2
    }
    
    return results

# --- 主程序 ---
if __name__ == "__main__":
    # 连杆长度参数
    R1 = 3.0
    R2 = 2.0
    
    # *** 水球的完整运动状态 (来自您的图片) ***
    P_c_input = (-1.389, 2.079)
    V_c_input = (9.028, 6.032)
    A_c_input = (16.898, -45.420)

    print("--- 输入参数 ---")
    print(f"R1 = {R1}, R2 = {R2}")
    print(f"水球位置 P_c: {P_c_input} m")
    print(f"水球速度 V_c: {V_c_input} m/s")
    print(f"水球加速度 A_c: {A_c_input} m/s^2\n")

    # 求解运动学
    solution = solve_general_kinematics(R1, R2, P_c_input, V_c_input, A_c_input)

    if solution:
        print("--- 求解结果 ---")
        # 对角度进行标准化，使其在 (-180, 180] 范围内
        theta_1_norm = (solution['theta_1_deg'] + 180) % 360 - 180
        theta_2_norm = (solution['theta_2_deg'] + 180) % 360 - 180

        print(f"角度 theta_1: {theta_1_norm:.2f} 度")
        print(f"角度 theta_2: {theta_2_norm:.2f} 度")
        print("-" * 20)
        print(f"角速度 omega_1: {solution['omega_1_rad_s']:.4f} rad/s")
        print(f"角速度 omega_2: {solution['omega_2_rad_s']:.4f} rad/s")
        print("-" * 20)
        print(f"角加速度 alpha_1: {solution['alpha_1_rad_s2']:.4f} rad/s^2")
        print(f"角加速度 alpha_2: {solution['alpha_2_rad_s2']:.4f} rad/s^2")
        R3 = 2.5
        G = 0 + 1.5j 

        # 2. 从 solution 字典中获取所需的运动学变量
        theta_2_deg = theta_2_norm
        omega_2 = solution['omega_2_rad_s']
        alpha_2 = solution['alpha_2_rad_s2']

        # 3. 将角度从度转换为弧度
        theta_2_rad = np.deg2rad(theta_2_deg)

        # 4. 使用复数进行计算
        # 创建表示方向的单位复数 e^(j*theta)
        unit_vector_complex = np.cos(theta_2_rad) + 1j * np.sin(theta_2_rad)

        # 计算位置矢量 r
        r_complex = G + R3 * unit_vector_complex

        # 计算速度矢量 v = j*w*r
        v_complex = 1j * omega_2 * (R3 * unit_vector_complex)

        # 计算加速度矢量 a = j*alpha*r - w^2*r
        a_complex = (1j * alpha_2 * (R3 * unit_vector_complex)) - (omega_2**2 * (R3 * unit_vector_complex))

        # 5. 打印结果
        print(f"位置 r (x, y):    ({r_complex.real:.4f}, {r_complex.imag:.4f}) m")
        print(f"速度 v (vx, vy):  ({v_complex.real:.4f}, {v_complex.imag:.4f}) m/s")
        print(f"加速度 a (ax, ay): ({a_complex.real:.4f}, {a_complex.imag:.4f}) m/s^2")

