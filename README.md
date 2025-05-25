# Segment Intersection Application

A Python application for visualizing and calculating intersection points of line segments in 2D space.

## Features

- Interactive graphical interface for segment manipulation
- Real-time intersection detection
- Support for:
  - Point intersections
  - Overlapping segments
  - Parallel segments
  - Collinear segments
- Dynamic coordinate grid with adaptive scaling
- Pan and zoom functionality
- Dark mode interface

## Requirements

- Python 3.x
- Tkinter (usually included with Python)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/segment-intersect.git
cd segment-intersect
```

2. Run the application:
```bash
python main.py
```

## Usage

### Adding Segments

Two methods are available:

1. **Manual Input**:
   - Enter coordinates in the input fields (X1, Y1, X2, Y2)
   - Click "Add" or press Enter

2. **Mouse Drawing**:
   - Click and drag on the canvas
   - Release to create segment

### View Controls

- **Zoom**: Use "Zoom +" and "Zoom -" buttons
- **Pan**: Hold middle mouse button and drag
- **Reset View**: Application automatically centers on (0,0) at start

### Finding Intersections

- Intersections are automatically calculated when segments are added
- Click "Find Intersections" to manually recalculate
- Results are shown in:
  - Results panel (detailed information)
  - Canvas (visual representation)

## Algorithm Description

### Intersection Detection

The application uses two main algorithms:

1. **Point Orientation Test**:
   - Determines relative orientation of three points
   - Used for initial intersection detection
   - Handles special cases (collinearity)

2. **Line Intersection Calculation**:
   - Uses parametric line equations
   - Handles parallel and overlapping segments
   - Provides exact intersection coordinates

### Time Complexity

- Intersection detection: O(n²) where n is number of segments
- Each individual intersection test: O(1)
- Grid rendering: O(w×h) where w,h are canvas dimensions

## Code Structure

```
segment-intersect/
├── main.py           # Main application file
└── README.md         # Documentation
```

### Key Components

1. **Core Classes**:
   - `EventPoint`: Represents points in the sweep line algorithm
   - `Status`: Maintains active segments state
   - `Intersection`: Stores intersection information
   - `GeometryApp`: Main UI application class

2. **Geometric Algorithms**:
   - `detect_intersection()`: Determines if segments intersect
   - `find_intersection_point()`: Calculates exact intersection
   - `check_all_intersections()`: Processes all segment pairs

3. **UI Components**:
   - Interactive canvas for visualization
   - Input fields for coordinates
   - Results panel for intersection data
   - Control buttons for key operations

## Example Usage

```python
# Create two intersecting segments
segment1 = ((0, 0), (5, 5))  # Diagonal line
segment2 = ((0, 5), (5, 0))  # Crossing diagonal

# Find intersection
result = find_intersection_point(segment1, segment2)
# Returns: (2.5, 2.5)
```

## Error Handling

The application includes comprehensive error checking for:
- Invalid input values
- Non-numeric data
- Identical endpoints
- Degenerate segments

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- Mateusz Październik

## Acknowledgments

- Computational Geometry algorithms based on standard geometric principles
- UI design inspired by modern dark-mode applications