import os
import time
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import imageio  # Library for creating GIFs

class DepthImageLoader:
    def __init__(self, scaling_factor=250.0):
        self.scaling_factor = scaling_factor

    def load(self, file_path):
        try:
            # Load depth image using matplotlib
            depth_image = plt.imread(file_path)
            # Scale depth values
            depth_image *= self.scaling_factor
            return depth_image
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            return None

class PointCloudConverter:
    def __init__(self, h_fov=(-90, 90), v_fov=(-24.9, 2.0), d_range=(0, 100)):
        self.h_fov = h_fov  # Horizontal field of view in degrees
        self.v_fov = v_fov  # Vertical field of view in degrees
        self.d_range = d_range  # Depth range in meters

    def convert(self, depth_image):
        if depth_image is None:
            return None
        
        # Calculate horizontal and vertical angles in radians
        h_angles = np.deg2rad(np.linspace(self.h_fov[0], self.h_fov[1], depth_image.shape[1]))[np.newaxis, :]
        v_angles = np.deg2rad(np.linspace(self.v_fov[0], self.v_fov[1], depth_image.shape[0]))[:, np.newaxis]

        # Convert depth image to 3D point cloud (x, y, z)
        x = depth_image * np.sin(h_angles) * np.cos(v_angles)
        y = depth_image * np.cos(h_angles) * np.cos(v_angles)
        z = depth_image * np.sin(v_angles)

        # Filter out points based on valid depth range
        valid_indices = (depth_image >= self.d_range[0]) & (depth_image <= self.d_range[1])
        return np.column_stack((x[valid_indices], y[valid_indices], z[valid_indices]))

class PointCloudAnimator:
    def __init__(self, update_interval=0.25, gif_filename='animation.gif'):
        self.update_interval = update_interval  # Interval between frame updates
        self.gif_filename = gif_filename  # Output GIF filename
        self.frames = []  # List to store images for each frame

    def capture_frame(self, vis):
        # Capture a screen image from the Open3D visualizer
        image = vis.capture_screen_float_buffer(do_render=True)
        image = np.asarray(image)
        image = (255 * image).astype(np.uint8)  # Convert to uint8 for imageio
        self.frames.append(image)

    def create_gif(self):
        # Save captured frames as a GIF
        imageio.mimsave(self.gif_filename, self.frames, fps=1/self.update_interval)

    def animate(self, point_clouds):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().background_color = np.array([0, 0, 0])  # Set background color to black

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(point_clouds[0])  # Initialize with first point cloud
        vis.add_geometry(point_cloud)

        frame_index = 0
        last_update_time = time.time()

        while True:
            current_time = time.time()
            if current_time - last_update_time > self.update_interval:
                point_cloud.points = o3d.utility.Vector3dVector(point_clouds[frame_index])  # Update point cloud
                vis.update_geometry(point_cloud)
                frame_index = (frame_index + 1) % len(point_clouds)  # Loop through frames
                last_update_time = current_time
                self.capture_frame(vis)  # Capture current frame for GIF
            vis.poll_events()
            vis.update_renderer()
            if not vis.poll_events():
                break
        vis.destroy_window()
        self.create_gif()  # Create GIF after animation

class PointCloudProcessor:
    def __init__(self, directory, scaling_factor=250.0, h_fov=(-90, 90), v_fov=(-24.9, 2.0), d_range=(0, 100), update_interval=0.25, gif_filename='animation.gif'):
        self.directory = directory  # Directory containing depth images
        self.loader = DepthImageLoader(scaling_factor)
        self.converter = PointCloudConverter(h_fov, v_fov, d_range)
        self.animator = PointCloudAnimator(update_interval, gif_filename)

    def process_and_animate(self):
        point_clouds = []
        for filename in sorted(os.listdir(self.directory)):
            if filename.endswith('.png'):
                file_path = os.path.join(self.directory, filename)
                depth_image = self.loader.load(file_path)
                if depth_image is not None:
                    point_cloud = self.converter.convert(depth_image)
                    if point_cloud is not None:
                        point_clouds.append(point_cloud)
        if point_clouds:
            self.animator.animate(point_clouds)  # Animate valid point clouds
        else:
            print("No valid point clouds generated.")

# Path to the directory containing depth images
directory = r'C:\Users\Yassine\Desktop\velodyne_points\depth_images'

# Initialize the processor and animate the point clouds
processor = PointCloudProcessor(directory)
processor.process_and_animate()
