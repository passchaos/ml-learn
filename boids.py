import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

class Boid:
    def __init__(self, position, velocity, max_speed=2.0, max_force=0.03):
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.max_speed = max_speed
        self.max_force = max_force
        self.perception_radius = 50.0
        self.separation_weight = 1.5
        self.alignment_weight = 1.0
        self.cohesion_weight = 1.0

    def update(self, boids):
        # 计算邻居
        neighbors = []
        for other in boids:
            if other is not self:
                distance = np.linalg.norm(self.position - other.position)
                if distance < self.perception_radius:
                    neighbors.append(other)

        # 应用三个规则
        separation = self.separate(neighbors)
        alignment = self.align(neighbors)
        cohesion = self.cohere(neighbors)

        # 应用权重
        separation *= self.separation_weight
        alignment *= self.alignment_weight
        cohesion *= self.cohesion_weight

        # 将力添加到速度
        self.velocity += separation + alignment + cohesion

        # 限制最大速度
        if np.linalg.norm(self.velocity) > self.max_speed:
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * self.max_speed

        # 更新位置
        self.position += self.velocity

        # 边界处理：如果超出边界则反弹
        self.boundary_check()

    def separate(self, neighbors):
        steer = np.array([0.0, 0.0])
        if len(neighbors) > 0:
            for neighbor in neighbors:
                diff = self.position - neighbor.position
                distance = np.linalg.norm(diff)
                if distance > 0 and distance < 25:  # 分离距离阈值
                    steer += diff / distance  # 向量归一化
            if np.linalg.norm(steer) > 0:
                steer = steer / np.linalg.norm(steer) * self.max_speed
                steer = steer - self.velocity
                if np.linalg.norm(steer) > self.max_force:
                    steer = steer / np.linalg.norm(steer) * self.max_force
        return steer

    def align(self, neighbors):
        avg_velocity = np.array([0.0, 0.0])
        if len(neighbors) > 0:
            for neighbor in neighbors:
                avg_velocity += neighbor.velocity
            avg_velocity /= len(neighbors)
            avg_velocity = avg_velocity / np.linalg.norm(avg_velocity) * self.max_speed
            steer = avg_velocity - self.velocity
            if np.linalg.norm(steer) > self.max_force:
                steer = steer / np.linalg.norm(steer) * self.max_force
            return steer
        return np.array([0.0, 0.0])

    def cohere(self, neighbors):
        center_of_mass = np.array([0.0, 0.0])
        if len(neighbors) > 0:
            for neighbor in neighbors:
                center_of_mass += neighbor.position
            center_of_mass /= len(neighbors)

            # 计算到中心点的方向向量
            desired = center_of_mass - self.position
            distance = np.linalg.norm(desired)
            if distance > 0:
                desired = desired / distance * self.max_speed
                steer = desired - self.velocity
                if np.linalg.norm(steer) > self.max_force:
                    steer = steer / np.linalg.norm(steer) * self.max_force
                return steer
        return np.array([0.0, 0.0])

    def boundary_check(self):
        # 简单的边界检查，如果超出画布边界则反弹
        if self.position[0] < 0:
            self.position[0] = 0
            self.velocity[0] *= -1
        elif self.position[0] > 800:
            self.position[0] = 800
            self.velocity[0] *= -1

        if self.position[1] < 0:
            self.position[1] = 0
            self.velocity[1] *= -1
        elif self.position[1] > 600:
            self.position[1] = 600
            self.velocity[1] *= -1

class BoidsSimulation:
    def __init__(self, num_boids=50):
        self.num_boids = num_boids
        self.boids = []
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim(0, 800)
        self.ax.set_ylim(0, 600)
        self.ax.set_title("Boids Algorithm Simulation")
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")

        # 初始化boids
        self.initialize_boids()

        # 初始化绘图对象
        self.scatter = self.ax.scatter([], [], s=10, c='blue')

    def initialize_boids(self):
        for _ in range(self.num_boids):
            position = [random.uniform(0, 800), random.uniform(0, 600)]

            v_value = 20
            velocity = [random.uniform(-v_value, v_value), random.uniform(-v_value, v_value)]
            self.boids.append(Boid(position, velocity))

    def update_frame(self, frame):
        # 更新每个boid
        for boid in self.boids:
            boid.update(self.boids)

        # 更新绘图数据
        positions = np.array([boid.position for boid in self.boids])
        self.scatter.set_offsets(positions)

        return self.scatter,

    def animate(self, interval=50):
        anim = FuncAnimation(
            self.fig,
            self.update_frame,
            frames=None,
            interval=interval,
            blit=True,
            repeat=True
        )
        plt.show()
        return anim

# 使用示例
if __name__ == "__main__":
    # 创建模拟
    simulation = BoidsSimulation(num_boids=500)

    # 开始动画
    animation = simulation.animate(interval=10)
