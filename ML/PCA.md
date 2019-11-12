## PCA

Principal Component Analysis



## Data reduction

Suppose there are points in the three-dimensional space. These points are distributed on an inclined plane that passes through the origin. 

<img src='https://github.com/daren996/PaperReading/blob/master/ML/Img/PCS-PLANE.png' style="width:80%">

#### Natural Coordinate System

If you use the three axes of the natural coordinate system x, y, and z to represent the set of data, you need to use three dimensions. However, the distribution of these points is only on a two-dimensional plane. 

#### Rotated Coordinate System

If the rotated coordinate system is denoted as x', y', z', then the representation of the set of data can only be represented by the two dimensions x' and y'. To recover the original representation, we can save the transformation matrix between these two coordinates. 

#### Linear Dependent

If the data is arranged in a matrix by row, then the rank of this matrix is 2. There is a correlation between these data, and the largest linearly independent group of vectors over the origin of these data contains 2 vectors, if the plane is assumed to pass the origin.  

#### Data Centering

What if the plane does not pass the origin? Data centering! Translate the origin of the coordinates to the data center so that the originally unrelated data is relevant in this new coordinate system. Interestingly, as three points must be coplanar, any three points after cemtering in the three-dimensional space are linearly related. 