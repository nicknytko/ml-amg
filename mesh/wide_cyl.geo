// Gmsh project created on Tue May 18 22:47:43 2021
SetFactory("OpenCASCADE");
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {0, 4, 0.6, 1.0};
//+
Recursive Delete {
  Point{2}; 
}
//+
Point(2) = {6, 0, 0, 1.0};
//+
Point(3) = {0, 2, 0, 1.0};
//+
Point(4) = {6, 2, 0, 1.0};
//+
Circle(1) = {1, 1, 0, 0.15, 0, 2*Pi};
//+
Line(2) = {3, 1};
//+
Line(3) = {1, 2};
//+
Line(4) = {2, 4};
//+
Line(5) = {4, 3};
//+
Physical Curve("Inlet", 6) = {2};
//+
Physical Curve("Oulet", 7) = {4};
//+
Physical Curve("Hole", 8) = {1};
//+
Physical Curve("Top Wall", 9) = {5};
//+
Physical Curve("Bottom Wall", 10) = {3};
//+
Curve Loop(1) = {2, 3, 4, 5};
//+
Curve Loop(2) = {1};
//+
Plane Surface(1) = {1, 2};
//+
Physical Surface("Domain", 11) = {1};
