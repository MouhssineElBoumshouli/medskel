import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from skimage import measure
import cv2
from pathlib import Path
import argparse

class ShapeSkeletonizer:
    """
    A class for extracting the skeleton of a shape using the bisector method.
    The algorithm follows these steps:
    1. Cartesianize (polygonize) the shape
    2. Calculate bisectors at vertices
    3. Find intersections of bisectors
    4. Connect the skeletal nodes
    """
    
    def __init__(self, cartesianization_level=50, pruning_threshold=10):
        """
        Initialize the skeletonizer with parameters.
        
        Args:
            cartesianization_level (int): Number of points to use when discretizing the shape
            pruning_threshold (float): Threshold for pruning small branches
        """
        self.cartesianization_level = cartesianization_level
        self.pruning_threshold = pruning_threshold
        self.vertices = None
        self.skeleton_nodes = None
        self.skeleton_edges = None
    
    def load_image(self, image_path):
        """Load an image and convert it to a binary mask."""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Threshold the image to get a binary mask
        _, binary_mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return binary_mask
    
    def cartesianize_shape(self, binary_mask):
        """
        Extract the boundary of the shape and convert it to a polygon.
        This corresponds to the Cartesianization step in the paper.
        """
        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            raise ValueError("No contours found in the image")
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Simplify the contour based on the cartesianization level
        epsilon = 0.01 * cv2.arcLength(largest_contour, True) * (100 / self.cartesianization_level)
        vertices = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Convert to a more manageable format
        self.vertices = np.array([point[0] for point in vertices])
        return self.vertices
    
    def calculate_bisectors(self):
        """
        Calculate bisectors for each vertex in the polygon.
        For each vertex i, calculate the bisector of the angle formed by 
        vertices i-1, i, and i+1.
        """
        num_vertices = len(self.vertices)
        bisectors = []
        
        for i in range(num_vertices):
            prev_idx = (i - 1) % num_vertices
            next_idx = (i + 1) % num_vertices
            
            # Get vertices that form the angle
            prev_vertex = self.vertices[prev_idx]
            current_vertex = self.vertices[i]
            next_vertex = self.vertices[next_idx]
            
            # Calculate vectors from current to neighboring vertices
            v1 = prev_vertex - current_vertex
            v2 = next_vertex - current_vertex
            
            # Normalize vectors
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            
            # Calculate bisector vector
            bisector = v1 + v2
            if np.linalg.norm(bisector) < 1e-10:
                # If the angle is 180 degrees, take perpendicular to one of the sides
                bisector = np.array([-v1[1], v1[0]])
            else:
                bisector = bisector / np.linalg.norm(bisector)
            
            # Make the bisector point inside the polygon
            # Calculate the centroid of the polygon
            centroid = np.mean(self.vertices, axis=0)
            
            # Make sure bisector points toward centroid
            if np.dot(bisector, centroid - current_vertex) < 0:
                bisector = -bisector
            
            # Store the bisector as a line (point, direction)
            bisectors.append((current_vertex, bisector))
        
        return bisectors
    
    def find_bisector_intersections(self, bisectors):
        """
        Find the intersections of bisectors within the shape.
        These intersections form the nodes of the skeleton.
        """
        num_bisectors = len(bisectors)
        skeleton_nodes = []
        
        # Calculate the centroid for checking if points are inside the polygon
        centroid = np.mean(self.vertices, axis=0)
        
        for i in range(num_bisectors):
            for j in range(i + 1, num_bisectors):
                # Get the bisector lines
                p1, v1 = bisectors[i]
                p2, v2 = bisectors[j]
                
                # Check if the lines are nearly parallel
                if abs(np.cross(v1, v2)) < 1e-10:
                    continue
                
                # Calculate intersection using the parametric form of the lines
                # p1 + t1*v1 = p2 + t2*v2
                # Solve for t1 and t2
                A = np.column_stack((v1, -v2))
                b = p2 - p1
                
                try:
                    t1, t2 = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    continue
                
                # Check if both parameters are positive (intersection is in the direction of the bisectors)
                if t1 > 0 and t2 > 0:
                    intersection = p1 + t1 * v1
                    
                    # Check if the intersection is inside the polygon
                    if self.point_in_polygon(intersection):
                        skeleton_nodes.append(intersection)
        
        self.skeleton_nodes = np.array(skeleton_nodes)
        return self.skeleton_nodes
    
    def point_in_polygon(self, point):
        """Check if a point is inside the polygon using the ray casting algorithm."""
        # Implementation of the ray casting algorithm
        x, y = point
        n = len(self.vertices)
        inside = False
        
        p1x, p1y = self.vertices[0]
        for i in range(1, n + 1):
            p2x, p2y = self.vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def connect_skeleton_nodes(self):
        """
        Connect the skeleton nodes to form the complete skeleton.
        The connection strategy follows the paper's guidelines.
        """
        if self.skeleton_nodes is None or len(self.skeleton_nodes) < 2:
            return []
        
        # Use Delaunay triangulation or MST to connect nodes
        # For simplicity, we'll use a proximity-based approach here
        
        edges = []
        threshold = np.max([np.linalg.norm(self.vertices[i] - self.vertices[(i+1) % len(self.vertices)]) 
                           for i in range(len(self.vertices))]) * 0.5
        
        for i in range(len(self.skeleton_nodes)):
            for j in range(i + 1, len(self.skeleton_nodes)):
                node1 = self.skeleton_nodes[i]
                node2 = self.skeleton_nodes[j]
                
                # Calculate the distance between nodes
                distance = np.linalg.norm(node1 - node2)
                
                # Connect nodes that are close enough
                if distance < threshold:
                    # Check if the connection is valid (stays inside the polygon)
                    valid = True
                    for t in np.linspace(0, 1, 10):
                        point = node1 * (1 - t) + node2 * t
                        if not self.point_in_polygon(point):
                            valid = False
                            break
                    
                    if valid:
                        edges.append((i, j))
        
        self.skeleton_edges = edges
        return edges

    def prune_branches(self):
        """Prune small branches from the skeleton."""
        if not self.skeleton_edges or not self.skeleton_nodes.any():
            return
        
        # Count connections for each node
        connections = {i: 0 for i in range(len(self.skeleton_nodes))}
        for edge in self.skeleton_edges:
            connections[edge[0]] += 1
            connections[edge[1]] += 1
        
        # Identify end points (nodes with only one connection)
        end_points = [i for i, count in connections.items() if count == 1]
        
        # For each end point, check if the branch is too short
        edges_to_remove = []
        for end_point in end_points:
            # Find the connected edge
            connected_edges = [edge for edge in self.skeleton_edges 
                              if end_point in edge]
            
            if not connected_edges:
                continue
                
            edge = connected_edges[0]
            other_node = edge[0] if edge[1] == end_point else edge[1]
            
            # Check the length of the branch
            branch_length = np.linalg.norm(
                self.skeleton_nodes[end_point] - self.skeleton_nodes[other_node]
            )
            
            if branch_length < self.pruning_threshold:
                edges_to_remove.append(edge)
        
        # Remove the edges
        self.skeleton_edges = [edge for edge in self.skeleton_edges 
                              if edge not in edges_to_remove]
    
    def skeletonize(self, input_data):
        """
        Main method to skeletonize a shape.
        
        Args:
            input_data: Can be a path to an image file, a binary mask, or a set of vertices
        
        Returns:
            tuple: (skeleton_nodes, skeleton_edges)
        """
        # Check the type of input data
        if isinstance(input_data, str) or isinstance(input_data, Path):
            # Load image from file path
            binary_mask = self.load_image(input_data)
            self.cartesianize_shape(binary_mask)
        elif isinstance(input_data, np.ndarray) and input_data.ndim == 2:
            # Binary mask provided
            self.cartesianize_shape(input_data)
        elif isinstance(input_data, np.ndarray) and input_data.ndim == 1:
            # Vertices provided
            self.vertices = input_data
        else:
            raise ValueError("Invalid input data format")
        
        # Calculate bisectors
        bisectors = self.calculate_bisectors()
        
        # Find intersections
        self.find_bisector_intersections(bisectors)
        
        # Connect nodes
        self.connect_skeleton_nodes()
        
        # Prune small branches
        self.prune_branches()
        
        return self.skeleton_nodes, self.skeleton_edges
    
    def visualize(self, show_vertices=True, show_skeleton=True, save_path=None):
        """
        Visualize the shape and its skeleton.
        
        Args:
            show_vertices (bool): Whether to show the vertices of the polygon
            show_skeleton (bool): Whether to show the skeleton
            save_path (str): Path to save the visualization
        """
        plt.figure(figsize=(10, 10))
        
        # Draw the polygon
        if self.vertices is not None and len(self.vertices) > 2:
            vertices_loop = np.vstack([self.vertices, self.vertices[0]])
            plt.plot(vertices_loop[:, 0], vertices_loop[:, 1], 'k-', linewidth=2, label='Shape Boundary')
            
            if show_vertices:
                plt.scatter(self.vertices[:, 0], self.vertices[:, 1], color='blue', s=50, label='Vertices')
        
        # Draw the skeleton
        if show_skeleton and self.skeleton_nodes is not None and self.skeleton_edges is not None:
            # Draw nodes
            plt.scatter(self.skeleton_nodes[:, 0], self.skeleton_nodes[:, 1], 
                        color='red', s=30, label='Skeleton Nodes')
            
            # Draw edges
            for edge in self.skeleton_edges:
                node1 = self.skeleton_nodes[edge[0]]
                node2 = self.skeleton_nodes[edge[1]]
                plt.plot([node1[0], node2[0]], [node1[1], node2[1]], 
                         'r-', linewidth=1.5)
        
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.title('Shape and Skeleton Visualization')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def main():
    """Command-line interface for the skeletonizer."""
    parser = argparse.ArgumentParser(description='Extract the skeleton from a shape.')
    parser.add_argument('input', type=str, help='Path to input image or shape file')
    parser.add_argument('--cartesianization-level', type=int, default=50, 
                        help='Level of detail for cartesianization (default: 50)')
    parser.add_argument('--pruning-threshold', type=float, default=10, 
                        help='Threshold for pruning small branches (default: 10)')
    parser.add_argument('--output', type=str, help='Path to save output visualization')
    parser.add_argument('--hide-vertices', action='store_true', help='Hide vertices in visualization')
    parser.add_argument('--hide-skeleton', action='store_true', help='Hide skeleton in visualization')
    
    args = parser.parse_args()
    
    # Create the skeletonizer
    skeletonizer = ShapeSkeletonizer(
        cartesianization_level=args.cartesianization_level,
        pruning_threshold=args.pruning_threshold
    )
    
    # Process the input
    nodes, edges = skeletonizer.skeletonize(args.input)
    
    # Visualize the result
    skeletonizer.visualize(
        show_vertices=not args.hide_vertices,
        show_skeleton=not args.hide_skeleton,
        save_path=args.output
    )
    
    print(f"Extracted skeleton with {len(nodes)} nodes and {len(edges)} connections.")


if __name__ == "__main__":
    main()