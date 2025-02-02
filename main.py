import os
import heapq
import tkinter as tk
from tkinter import messagebox

os.environ['TK_SILENCE_DEPRECATION'] = '1'

# Core geometric classes
class EventPoint:
    """Event point class (upper/lower segment endpoint or intersection point)"""
    def __init__(self, x, y, segment=None, is_upper=False, is_lower=False):
        self.x = x
        self.y = y
        self.segment = segment
        self.is_upper = is_upper
        self.is_lower = is_lower

    def __lt__(self, other):
        return (self.y, self.x) < (other.y, other.x)

class Status:
    """
    Maintains active segments intersecting with sweep line.
    Segments are sorted by their intersection point with sweep line.
    """
    def __init__(self):
        self.segments = []  # list of active segments
        self.segment_index = {}  # mapping segments to their indices
    
    def _normalize_segment(self, segment):
        """Normalizes segment representation to a unique form"""
        (x1, y1), (x2, y2) = segment
        if y1 > y2 or (y1 == y2 and x1 > x2):
            return ((x2, y2), (x1, y1))
        return segment
    
    def add(self, segment):
        normalized = self._normalize_segment(segment)
        if normalized not in self.segment_index:
            self.segments.append(segment)
            # Sort by intersection with sweep line
            sweep_y = segment[0][1]  # use y-coordinate of upper point
            self.segments.sort(key=lambda s: self._get_x_at_y(s, sweep_y))
            self.segment_index[normalized] = self.segments.index(segment)
    
    def _get_x_at_y(self, segment, y):
        """Calculates x-coordinate of intersection with horizontal line y"""
        (x1, y1), (x2, y2) = segment
        if y1 == y2:
            return x1
        t = (y - y1) / (y2 - y1)
        return x1 + t * (x2 - x1)
    
    def remove(self, segment):
        normalized = self._normalize_segment(segment)
        if normalized in self.segment_index:
            index = self.segment_index.pop(normalized)
            self.segments.pop(index)
            # Update indices
            for i in range(index, len(self.segments)):
                self.segment_index[self._normalize_segment(self.segments[i])] = i
    
    def find_neighbors(self, segment):
        normalized = self._normalize_segment(segment)
        if normalized not in self.segment_index:
            return None, None
        index = self.segment_index[normalized]
        left_neighbor = self.segments[index - 1] if index > 0 else None
        right_neighbor = self.segments[index + 1] if index < len(self.segments) - 1 else None
        return left_neighbor, right_neighbor

# Geometric algorithms
def detect_intersection(segment1, segment2):
    """
    Checks if two segments intersect using point orientation.
    Handles special cases like collinear points.
    """
    # Handle case when segments are identical
    if segment1 == segment2:
        return True

    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        eps = 1e-10  # tolerance for rounding errors
        if abs(val) < eps:
            return 0
        return 1 if val > 0 else 2

    def on_segment(p, q, r):
        if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]):
            return True
        return False

    p1, q1 = segment1
    p2, q2 = segment2

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, p2, q1):
        return True

    if o2 == 0 and on_segment(p1, q2, q1):
        return True

    if o3 == 0 and on_segment(p2, p1, q2):
        return True

    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False

def find_intersection_point(segment1, segment2):
    """Calculates intersection point or common segment"""
    (x1, y1), (x2, y2) = segment1
    (x3, y3), (x4, y4) = segment2
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    EPSILON = 1e-10
    
    if abs(denom) < EPSILON:  # Parallel segments
        if detect_intersection(segment1, segment2):
            # Find common segment
            points = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], 
                          key=lambda p: (p[0], p[1]))
            # Check if segments actually overlap
            if points[1] == points[2]:  # Touch at a single point
                return points[1]
            # Return start and end points of common section
            return (points[1], points[2])
        return None
    
    # Calculate intersection point
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    
    if 0 <= t <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (x2 - x1)  # Fixed: using correct y increment
        
        # Verify point lies on second segment
        if abs(x4 - x3) > EPSILON:
            u = (x - x3) / (x4 - x3)
        elif abs(y4 - y3) > EPSILON:
            u = (y - y3) / (y4 - y3)
        else:
            return None
            
        if 0 <= u <= 1:
            return (x, y)
    return None

class Intersection:
    """
    Stores intersection information:
    - intersection point (if single point)
    - participating segments
    - intersection type (point/overlap)
    """
    def __init__(self, point=None, segment1=None, segment2=None, is_overlap=False, overlap_points=None):
        self.point = point  # (x, y) for intersection point
        self.segment1 = segment1
        self.segment2 = segment2
        self.is_overlap = is_overlap
        self.overlap_points = overlap_points  # (start, end) for overlapping

def handle_event_point(event_point, status, event_queue, intersections):
    try:
        print(f"\nHandling event point: ({event_point.x}, {event_point.y})")
        print(f"Point type: {'upper' if event_point.is_upper else 'lower' if event_point.is_lower else 'intersection'}")
        
        if event_point.is_upper:
            status.add(event_point.segment)
            print(f"Added segment: {event_point.segment}")
            left_neighbor, right_neighbor = status.find_neighbors(event_point.segment)
            print(f"Neighbors: left={left_neighbor}, right={right_neighbor}")
            
            if left_neighbor and detect_intersection(left_neighbor, event_point.segment):
                intersection_point = find_intersection_point(left_neighbor, event_point.segment)
                print(f"Found intersection with left neighbor: {intersection_point}")
                if intersection_point:
                    if intersection_point == "overlap":
                        intersections.append(Intersection(
                            segment1=left_neighbor,
                            segment2=event_point.segment,
                            is_overlap=True
                        ))
                    else:
                        intersections.append(Intersection(
                            point=intersection_point,
                            segment1=left_neighbor,
                            segment2=event_point.segment
                        ))
                        heapq.heappush(event_queue, EventPoint(intersection_point[0], intersection_point[1]))
            if right_neighbor and detect_intersection(right_neighbor, event_point.segment):
                intersection_point = find_intersection_point(right_neighbor, event_point.segment)
                print(f"Found intersection with right neighbor: {intersection_point}")
                if intersection_point:
                    if intersection_point == "overlap":
                        intersections.append(Intersection(
                            segment1=right_neighbor,
                            segment2=event_point.segment,
                            is_overlap=True
                        ))
                    else:
                        intersections.append(Intersection(
                            point=intersection_point,
                            segment1=right_neighbor,
                            segment2=event_point.segment
                        ))
                        heapq.heappush(event_queue, EventPoint(intersection_point[0], intersection_point[1]))
        elif event_point.is_lower:
            left_neighbor, right_neighbor = status.find_neighbors(event_point.segment)
            status.remove(event_point.segment)
            if left_neighbor and right_neighbor and detect_intersection(left_neighbor, right_neighbor):
                intersection_point = find_intersection_point(left_neighbor, right_neighbor)
                if intersection_point:
                    if intersection_point == "overlap":
                        intersections.append(Intersection(
                            segment1=left_neighbor,
                            segment2=right_neighbor,
                            is_overlap=True
                        ))
                    else:
                        intersections.append(Intersection(
                            point=intersection_point,
                            segment1=left_neighbor,
                            segment2=right_neighbor
                        ))
                        heapq.heappush(event_queue, EventPoint(intersection_point[0], intersection_point[1]))
    except KeyError as e:
        print(f"Error processing segment: {e}")

def input_segments():
    """Reads input data from user"""
    segments = []
    n = int(input("Enter the number of segments: "))
    for _ in range(n):
        x1, y1 = map(float, input("Enter the coordinates of the first point (x1 y1): ").split())
        x2, y2 = map(float, input("Enter the coordinates of the second point (x2 y2): ").split())
        segments.append(((x1, y1), (x2, y2)))
    return segments

# Helper functions for visualization and displaying results
def display_segments(segments):
    """Displays visualization of segments in ASCII art"""
    print("\nSegments visualization:")
    
    # Find dimensions of visualization area
    min_x = min(min(x1, x2) for (x1, y1), (x2, y2) in segments)
    max_x = max(max(x1, x2) for (x1, y1), (x2, y2) in segments)
    min_y = min(min(y1, y2) for (x1, y1), (x2, y2) in segments)
    max_y = max(max(y1, y2) for (x1, y1), (x2, y2) in segments)
    
    # Create grid
    width = int(max_x - min_x) + 3
    height = int(max_y - min_y) + 3
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Draw axes
    for i in range(height):
        grid[i][0] = '|'
    for j in range(width):
        grid[height-1][j] = '-'
    grid[height-1][0] = '+'
    
    # Draw segments
    for idx, segment in enumerate(segments):
        (x1, y1), (x2, y2) = segment
        # Scale coordinates to grid
        grid_x1 = int(x1 - min_x + 1)
        grid_y1 = int(height - (y1 - min_y + 2))
        grid_x2 = int(x2 - min_x + 1)
        grid_y2 = int(height - (y2 - min_y + 2))
        
        # Draw segment using segment number symbol
        symbol = str(idx + 1)
        # Bresenham's algorithm for drawing lines
        dx = abs(grid_x2 - grid_x1)
        dy = abs(grid_y2 - grid_y1)
        x, y = grid_x1, grid_y1
        step_x = 1 if grid_x2 > grid_x1 else -1
        step_y = 1 if grid_y2 > grid_y1 else -1
        
        if dx > dy:
            err = dx / 2
            while x != grid_x2:
                if 0 <= x < width and 0 <= y < height:
                    grid[y][x] = symbol
                err -= dy
                if err < 0:
                    y += step_y
                    err += dx
                x += step_x
        else:
            err = dy / 2
            while y != grid_y2:
                if 0 <= x < width and 0 <= y < height:
                    grid[y][x] = symbol
                err -= dx
                if err < 0:
                    x += step_x
                    err += dy
                y += step_y
        
        if 0 <= grid_x2 < width and 0 <= grid_y2 < height:
            grid[grid_y2][grid_x2] = symbol
    
    # Display grid
    for row in grid:
        print(''.join(row))
    
    # Display legend
    print("\nLegend:")
    for i, segment in enumerate(segments):
        print(f"{i+1}: {segment}")

def display_intersections(intersections, segments):
    """Displays found intersection points and common segments"""
    if not intersections:
        print("\nNo intersections or common segments found.")
        return

    print("\nFound intersections and common segments:")
    for i, intersection in enumerate(intersections, 1):
        if intersection.is_overlap:
            print(f"\n{i}. Common segments:")
            print(f"   Segment 1: {intersection.segment1}")
            print(f"   Segment 2: {intersection.segment2}")
        else:
            print(f"\n{i}. Intersection point: {intersection.point}")
            print(f"   Between segments:")
            print(f"   Segment 1: {intersection.segment1}")
            print(f"   Segment 2: {intersection.segment2}")

# Function to check all possible intersections between segments
def check_all_intersections(segments):
    """
    Checks all possible pairs of segments for intersections.
    Complexity: O(n^2), where n is the number of segments.
    """
    intersections = []
    for i, seg1 in enumerate(segments):
        for j, seg2 in enumerate(segments[i+1:], i+1):
            if detect_intersection(seg1, seg2):
                result = find_intersection_point(seg1, seg2)
                if result:
                    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], tuple):
                        # Overlapping segments
                        intersections.append(Intersection(
                            segment1=seg1,
                            segment2=seg2,
                            is_overlap=True,
                            overlap_points=result
                        ))
                    else:
                        # Single intersection point
                        intersections.append(Intersection(
                            point=result,
                            segment1=seg1,
                            segment2=seg2
                        ))
    return intersections

# UI Class
class SegmentIntersectApp:
    """Main application class for segment intersection visualization"""
    def center_view(self):
        """Centers view on point (0,0)"""
        self.root.update_idletasks()  # Ensure window dimensions are updated
        w = self.canvas.winfo_width() if hasattr(self, 'canvas') else 600
        h = self.canvas.winfo_height() if hasattr(self, 'canvas') else 500
        self.offset_x = w // 2
        self.offset_y = h // 2

    def __init__(self, root):
        """Initialize application with UI components"""
        self.root = root
        self.root.title("Segment Intersect App")
        
        # Set minimum window size
        self.root.minsize(900, 600)
        
        # Dark theme colors
        self.colors = {
            'bg': '#1e1e1e',
            'frame_bg': '#2d2d2d',
            'text': '#ffffff',
            'entry_bg': '#3d3d3d',
            'entry_fg': '#ffffff',
            'button_bg': '#4d4d4d',
            'button_fg': '#ffffff',
            'canvas_bg': '#2d2d2d',
            'line_color': '#00ff00',
            'intersection_color': '#ff0000'
        }
        
        # Set colors for main window
        self.root.configure(bg=self.colors['bg'])
        
        # Variables
        self.segments = []
        self.temp_segment = None
        self.start_point = None
        self.scale_factor = 60.0  # Adjusted to fit the scale from 1 to 15
        self.grid_size = 50  # grid size
        self.entries = {}
        
        # Add variables for view panning
        self.pan_start_x = None
        self.pan_start_y = None
        self.is_panning = False
        
        # Create interface
        self.create_widgets()
        
        # Initialize view position after creating canvas
        self.center_view()

    # UI Methods
    def create_widgets(self):
        """Create and configure all UI elements"""
        # Main container without PanedWindow
        main_container = tk.Frame(self.root, bg=self.colors['bg'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel
        left_panel = tk.Frame(main_container, bg=self.colors['bg'], width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_panel.pack_propagate(False)

        # Data input panel - new layout
        input_frame = tk.LabelFrame(
            left_panel, 
            text="Coordinates Input", 
            bg=self.colors['frame_bg'], 
            fg=self.colors['text'],
            font=('Arial', 10, 'bold')
        )
        input_frame.pack(fill=tk.X, padx=5, pady=5)

        # Container for input fields (left side)
        fields_frame = tk.Frame(input_frame, bg=self.colors['frame_bg'])
        fields_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)

        # Grid for input fields
        for i, (label, var) in enumerate([('X1:', 'x1'), ('Y1:', 'y1'), ('X2:', 'x2'), ('Y2:', 'y2')]):
            tk.Label(
                fields_frame, 
                text=label, 
                bg=self.colors['frame_bg'], 
                fg=self.colors['text'],
                font=('Arial', 10)
            ).grid(row=i, column=0, padx=5, pady=5, sticky='e')
            
            entry = tk.Entry(
                fields_frame,
                bg=self.colors['entry_bg'],
                fg=self.colors['entry_fg'],
                insertbackground=self.colors['text'],
                width=10,
                font=('Arial', 10)
            )
            entry.grid(row=i, column=1, padx=5, pady=5, sticky='w')
            self.entries[var] = entry

        # Container for buttons (right side)
        button_frame = tk.Frame(input_frame, bg=self.colors['frame_bg'])
        button_frame.pack(side=tk.RIGHT, padx=5, pady=5, fill=tk.Y)

        buttons = [
            ("Add", lambda: self._safe_call(self.add_segment)),
            ("Clear", lambda: self._safe_call(self.clear_canvas)),
            ("Find Intersections", lambda: self._safe_call(self.find_intersections))
        ]

        for i, (text, cmd) in enumerate(buttons):
            btn = tk.Button(
                button_frame,
                text=text,
                command=cmd,
                bg=self.colors['button_bg'],
                fg=self.colors['button_fg'],
                font=('Arial', 10),
                width=12
            )
            btn.pack(pady=2)

        # Results list
        results_frame = tk.LabelFrame(
            left_panel, 
            text="Results", 
            bg=self.colors['frame_bg'],
            fg=self.colors['text'],
            font=('Arial', 10, 'bold')
        )
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add scrollbars to results list
        results_scroll = tk.Scrollbar(results_frame)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.results_list = tk.Listbox(
            results_frame,
            bg=self.colors['entry_bg'],
            fg=self.colors['text'],
            font=('Arial', 10),
            yscrollcommand=results_scroll.set
        )
        self.results_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scroll.config(command=self.results_list.yview)

        # Right panel with canvas
        right_panel = tk.Frame(main_container, bg=self.colors['bg'])
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(
            right_panel,
            bg=self.colors['canvas_bg'],
            width=600,
            height=500
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Update center when canvas is resized
        self.canvas.bind('<Configure>', self.on_canvas_configure)
        
        # Add zoom buttons
        zoom_frame = tk.Frame(right_panel, bg=self.colors['bg'])
        zoom_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Button(
            zoom_frame,
            text="Zoom +",
            command=self.zoom_in,
            bg=self.colors['button_bg'],
            fg=self.colors['button_fg'],
            font=('Arial', 10)
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            zoom_frame,
            text="Zoom -",
            command=self.zoom_out,
            bg=self.colors['button_bg'],
            fg=self.colors['button_fg'],
            font=('Arial', 10)
        ).pack(side=tk.LEFT, padx=5)

        # Bind mouse events
        self.canvas.bind("<Button-1>", self.start_segment)
        self.canvas.bind("<B1-Motion>", self.drag_segment)
        self.canvas.bind("<ButtonRelease-1>", self.end_segment)

        # Add view panning (middle mouse button)
        self.canvas.bind("<Button-2>", self.start_pan)  # Middle click
        self.canvas.bind("<B2-Motion>", self.pan)  # Middle click drag
        self.canvas.bind("<ButtonRelease-2>", self.end_pan)  # Middle click release

        # Add Enter key handling for input fields
        for entry in self.entries.values():
            entry.bind('<Return>', lambda e: self._safe_call(self.add_segment))

    def on_canvas_configure(self, event):
        """Handles canvas resize and centers view"""
        self.offset_x = event.width // 2
        self.offset_y = event.height // 2
        self.draw_grid()

    def _safe_call(self, func):
        """Safe function call with error handling and interface update"""
        try:
            func()
            self.draw_grid()  # Refresh grid
            self.root.update()  # Update interface
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def add_segment(self):
        """Adds new segment with data validation"""
        try:
            # Check if fields are not empty and are numbers
            fields = ['x1', 'y1', 'x2', 'y2']
            coords = {}
            
            for field in fields:
                value = self.entries[field].get().strip()
                if not value:
                    self.entries[field].config(bg='#ff6b6b')  # Highlight empty field
                    messagebox.showwarning("Warning", f"Field {field} is empty")
                    return
                try:
                    coords[field] = float(value)
                    self.entries[field].config(bg=self.colors['entry_bg'])  # Restore normal color
                except ValueError:
                    self.entries[field].config(bg='#ff6b6b')  # Highlight invalid field
                    messagebox.showerror("Error", f"Value in field {field} is not a number")
                    return
            
            # Check if points are not identical
            if (coords['x1'], coords['y1']) == (coords['x2'], coords['y2']):
                messagebox.showwarning("Warning", "Start and end points cannot be identical")
                return
            
            # Create and add segment
            segment = ((coords['x1'], coords['y1']), (coords['x2'], coords['y2']))
            self.segments.append(segment)
            
            # Draw segment
            self.canvas.create_line(
                coords['x1'] * self.scale_factor + self.offset_x,
                self.offset_y - coords['y1'] * self.scale_factor,  # Inverted Y
                coords['x2'] * self.scale_factor + self.offset_x,
                self.offset_y - coords['y2'] * self.scale_factor,  # Inverted Y
                fill=self.colors['line_color'],
                width=2
            )
            
            # Clear fields
            for entry in self.entries.values():
                entry.delete(0, tk.END)
                entry.config(bg=self.colors['entry_bg'])
            
            # Automatically find intersections and update view
            self.find_intersections()
            self.root.update()
            
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")

    def zoom_in(self):
        self.scale_factor *= 2
        self.redraw_canvas()

    def zoom_out(self):
        self.scale_factor /= 2
        self.redraw_canvas()

    def redraw_canvas(self):
        """Redraw all canvas elements"""
        self.canvas.delete("all")
        self.draw_grid()  # Draw grid first
        if self.segments:  # Only if there are segments
            # Draw all segments
            for segment in self.segments:
                (x1, y1), (x2, y2) = segment
                self.canvas.create_line(
                    x1 * self.scale_factor + self.offset_x, self.offset_y - y1 * self.scale_factor,  # Inverted Y
                    x2 * self.scale_factor + self.offset_x, self.offset_y - y2 * self.scale_factor,  # Inverted Y
                    fill=self.colors['line_color']
                )
            # Find and show intersections again, but without displaying message
            self._find_intersections_silent()

    def _find_intersections_silent(self):
        """Find intersections without displaying messages"""
        if not self.segments:
            return
        intersections = check_all_intersections(self.segments)
        self.results_list.delete(0, tk.END)
        if not intersections:
            self.results_list.insert(tk.END, "No intersections found")
            return
        for i, intersection in enumerate(intersections, 1):
            if intersection.is_overlap and intersection.overlap_points:
                start, end = intersection.overlap_points
                # Draw common segment
                self.canvas.create_line(
                    start[0] * self.scale_factor + self.offset_x,
                    self.offset_y - start[1] * self.scale_factor,
                    end[0] * self.scale_factor + self.offset_x,
                    self.offset_y - end[1] * self.scale_factor,
                    fill=self.colors['intersection_color'],
                    width=3
                )
                # Display start and end coordinates
                self.canvas.create_text(
                    (start[0] + end[0])/2 * self.scale_factor + self.offset_x,
                    self.offset_y - (start[1] + end[1])/2 * self.scale_factor - 15,
                    text=f"({start[0]:.1f}, {start[1]:.1f}) - ({end[0]:.1f}, {end[1]:.1f})",
                    fill=self.colors['intersection_color'],
                    font=('Arial', 8)
                )
                self.results_list.insert(tk.END, f"Intersection {i}: Common segment")
                self.results_list.insert(tk.END, f"  Start: ({start[0]:.2f}, {start[1]:.2f})")
                self.results_list.insert(tk.END, f"  End: ({end[0]:.2f}, {end[1]:.2f})")
                self.results_list.insert(tk.END, "")
            elif not intersection.is_overlap:
                x, y = intersection.point
                px = x * self.scale_factor + self.offset_x
                py = self.offset_y - y * self.scale_factor  # Inverted Y
                r = 4
                # Draw intersection point
                self.canvas.create_oval(
                    px - r, py - r,
                    px + r, py + r,
                    outline=self.colors['intersection_color'],
                    width=2
                )
                # Add text with coordinates above intersection point
                self.canvas.create_text(
                    px, py - 15,  # 15 pixels above the point
                    text=f"({x:.1f}, {y:.1f})",
                    fill=self.colors['intersection_color'],
                    font=('Arial', 8)
                )
                self.results_list.insert(tk.END, f"Intersection {i}: Point ({x:.2f}, {y:.2f})")
                self.results_list.insert(tk.END, f"  Segment 1: {intersection.segment1}")
                self.results_list.insert(tk.END, f"  Segment 2: {intersection.segment2}")
                self.results_list.insert(tk.END, "")

    def find_intersections(self):
        """Version with message, called by button"""
        if not self.segments:
            self.results_list.delete(0, tk.END)
            self.results_list.insert(tk.END, "No segments to check")
            return
        self._find_intersections_silent()

    def draw_grid(self):
        """Draw coordinate grid with adaptive scale"""
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        
        # Helper function to determine numbering step
        def get_numbering_step(start, end):
            visible_numbers = abs(end - start) + 1
            if visible_numbers <= 15:
                return 1
            elif visible_numbers <= 45:
                return 2
            elif visible_numbers <= 90:
                return 3
            else:
                return 3 + (visible_numbers - 90) // 30
        
        # Calculate range of visible numbers
        left_value = int((0 - self.offset_x) / self.scale_factor)
        right_value = int((w - self.offset_x) / self.scale_factor)
        top_value = int(-(0 - self.offset_y) / self.scale_factor)
        bottom_value = int(-(h - self.offset_y) / self.scale_factor)
        
        # Set numbering steps
        x_step = get_numbering_step(left_value, right_value)
        y_step = get_numbering_step(bottom_value, top_value)
        
        # Draw vertical grid lines
        for x in range(left_value, right_value + 1):
            if x % x_step == 0:
                px = x * self.scale_factor + self.offset_x
                if 0 <= px <= w:
                    # Lighter line for Y axis
                    color = '#ffffff' if x == 0 else '#404040'
                    width = 2 if x == 0 else 1
                    self.canvas.create_line(
                        px, 0, px, h,
                        fill=color,
                        width=width,
                        dash=(2, 4) if x != 0 else None
                    )
        
        # Draw horizontal grid lines
        for y in range(bottom_value, top_value + 1):
            if y % y_step == 0:
                py = self.offset_y - y * self.scale_factor
                if 0 <= py <= h:
                    # Lighter line for X axis
                    color = '#ffffff' if y == 0 else '#404040'
                    width = 2 if y == 0 else 1
                    self.canvas.create_line(
                        0, py, w, py,
                        fill=color,
                        width=width,
                        dash=(2, 4) if y != 0 else None
                    )
        
        # Draw values on OX axis
        for x in range(left_value, right_value + 1):
            if x % x_step == 0:
                px = x * self.scale_factor + self.offset_x
                if 0 <= px <= w:
                    self.canvas.create_text(
                        px, self.offset_y + 10,
                        text=str(x),
                        fill='#ffffff'
                    )
        
        # Draw values on OY axis
        for y in range(bottom_value, top_value + 1):
            if y % y_step == 0:
                py = self.offset_y - y * self.scale_factor
                if 0 <= py <= h:
                    self.canvas.create_text(
                        self.offset_x - 10, py,
                        text=str(y),
                        fill='#ffffff'
                    )

    def clear_canvas(self):
        self.canvas.delete("all")
        self.segments = []
        self.results_list.delete(0, tk.END)

    # Event Handlers
    def start_segment(self, event):
        """Start drawing new segment"""
        canvas_x = event.x
        canvas_y = event.y
        # Convert canvas coordinates to mathematical coordinates
        self.start_point = (
            (canvas_x - self.offset_x) / self.scale_factor,
            -(canvas_y - self.offset_y) / self.scale_factor  # Inverted Y
        )
        self.temp_segment = self.canvas.create_line(
            canvas_x, canvas_y, canvas_x, canvas_y,
            fill=self.colors['line_color']
        )

    def drag_segment(self, event):
        """Update temporary segment during drag"""
        if self.temp_segment:
            self.canvas.coords(
                self.temp_segment,
                self.start_point[0] * self.scale_factor + self.offset_x,
                self.offset_y - self.start_point[1] * self.scale_factor,  # Inverted Y
                event.x,
                event.y
            )
    
    def end_segment(self, event):
        """Finish segment drawing and add to list"""
        if self.temp_segment:
            canvas_x = event.x
            canvas_y = event.y
            # Convert canvas coordinates to mathematical coordinates
            end_point = (
                (canvas_x - self.offset_x) / self.scale_factor,
                -(canvas_y - self.offset_y) / self.scale_factor  # Inverted Y
            )
            self.segments.append((self.start_point, end_point))
            self.temp_segment = None
            self.start_point = None
            # Immediately find intersections
            self.find_intersections()

    def start_pan(self, event):
        """Start view panning"""
        self.canvas.config(cursor="fleur")  # Change cursor to "hand"
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.is_panning = True

    def pan(self, event):
        """Pan view during drag"""
        if not self.is_panning:
            return
            
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        
        self.offset_x += dx
        self.offset_y += dy
        
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        
        self.redraw_canvas()

    def end_pan(self, event):
        """End view panning"""
        self.canvas.config(cursor="")  # Restore default cursor
        self.is_panning = False
        self.pan_start_x = None
        self.pan_start_y = None

# Main entry point
def main():
    """Application entry point"""
    root = tk.Tk()
    app = SegmentIntersectApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
