#version 430 core
#extension GL_ARB_compute_shader : enable
#extension GL_ARB_shader_image_load_store : enable

// output of compute shader is a TEXTURE
layout(rgba32f, binding = 0) uniform image2D outTexture;


// receive as inputs, rays from the centre of the camera to the frustum corners
layout( location = 0 ) uniform vec4 ray00; // top left in world space
layout( location = 1 ) uniform vec4 ray01; // top right in world space
layout( location = 2 ) uniform vec4 ray10; // bottom left in world space
layout( location = 3 ) uniform vec4 ray11; // bottom right in world space

// receive the camera position
layout( location = 4 ) uniform vec3 camPos; // camera position

// receive how many primitives of each type
// geomCount.x = number of spheres
// geomCount.y = number of triangles
// geomCount.z = number of planes
// geomCount.w = number of obbs
layout( location = 5 ) uniform ivec4 geomCount; // sphere, triangle, planes, obbs

// how many lights to process (how many elements in the array "light")
layout( location = 6 ) uniform int lightCount;

// receive an ARRAY of light_type, 
struct light_type {
	 vec4 position;
	 vec4 colour;
};
layout(std430, binding = 2) buffer light
{
	light_type lights_data[];
};

struct sphere_type {
	vec4 centre_radius;
	vec4 colour;
};
// 8 floats per sphere: cx cy cz rad r g b a
layout(std430, binding = 3) buffer sphere
{
	sphere_type spheres_data[];
};

struct plane_type {
	vec4 normal_d;
	vec4 colour;
};
// 8 floats per plane: nx ny nz d r g b a
layout(std430, binding = 4) buffer plane
{
	plane_type planes_data[];
};

struct triangle_type {
	vec4 vtx0, vtx1, vtx2;
	vec4 colour;
};
layout(std430, binding = 5) buffer triangle
{
	triangle_type triangles_data[];
};

// 20 floats per obb: centre4, u3, hu, v3, hv, w3, hw, rgba4
struct obb_type {
	vec4 centre, u_hu, v_hv, w_hw, colour;
};
layout(std430, binding = 6) buffer obb
{
	obb_type obb_data[];
};

layout (local_size_x = 20, local_size_y = 20) in;

#define SPHERE_TYPE 0
#define LIGHT_TYPE 1
#define PLANE_TYPE 2
#define TRIANGLE_TYPE 3
#define OBB_TYPE 4

// primitives tests forward declarations
float sphereTest(vec3 rayDir, vec3 rayOrigin, vec4 centre_radius);
float planeTest(vec3 rayDir, vec3 rayOrigin, vec4 plane_info);
float triangleTest(vec3 rayDir, vec3 rayOrigin, triangle_type tri);
float obbTest(vec3 rayDir, vec3 rayOrigin, obb_type obb);

// entire scene test forward declaration
void sceneTest(in vec4 ray_dir, inout float lastT, inout int objIndex, inout int objType);

// shade a point of a surface (for any primitive) forward declaration
vec4 shade(in vec3 pointOnSurface, in vec3 normal, in vec3 colour);

vec4 castRay(ivec2 pos)
{
	// normalise values from [ 0,width; 0,height ] to [ 0,1; 0,1]
	vec2 interp = vec2(pos) / imageSize(outTexture); 

	// compute ray as interpolation of input rays (corners)
	vec4 ray_dir = normalize(
		mix(
			mix(ray00,ray01,interp.y), // left corners together
			mix(ray10,ray11,interp.y), // right corners together
			interp.x // join new interpolated rays into a final ray
		)
	);

	// lastT is the last intersection found (will keep the closest value)
	float lastT=-1;
	// will remember which object index was hit if any.
	int objIndex = -1;
	// will remember which object type was hit if any.
	int objType = -1;

	// do ray versus scene test
	sceneTest(ray_dir, lastT, objIndex, objType); 

	// set pixel to BACKGROUND colour
	vec4 finalPixelOut = vec4(0.2,0.2,0.55, 1.0);

	if (objIndex >= 0) {
		// did we hit something?

		// IMPLEMENT HERE.
		// FIND POINT IN SPACE WHERE THE INTERSECTION HAPPENED.
		vec3 pointOnSurface = camPos + lastT * ray_dir.xyz;

		if (objType==SPHERE_TYPE)
		{
			// IMPLEMENT HERE
			sphere_type sphere = spheres_data[objIndex];
			
			vec3 normal = normalize(pointOnSurface - sphere.centre_radius.xyz);
			vec3 objColor = sphere.colour.xyz;

			finalPixelOut = shade(pointOnSurface, normal,objColor);
			// COMPUTE FINAL PIXEL COLOUR USING SHADE() FUNCTION
		}
		else if (objType==LIGHT_TYPE)
		{
			// IMPLEMENT HERE.
			// LIGHTS ARE DRAWN AS SPHERES, JUST GIVE A CONSTANT COLOUR TO IT.
			finalPixelOut = vec4(1.0f,1.0f,1.0f,1.0f);
			// DO NOT SHADE WITH SHADE() FUNCTION.
		}
		else if (objType == PLANE_TYPE)
		{
			// IMPLEMENT HERE
			// COMPUTE FINAL PIXEL COLOUR USING SHADE() FUNCTION

			plane_type plane = planes_data[objIndex];
			vec3 objColor = plane.colour.xyz;
			finalPixelOut = shade(pointOnSurface, normalize(plane.normal_d.xyz),objColor);
		}
		else if (objType == TRIANGLE_TYPE) 
		{
			// IMPLEMENT HERE
			// Triangle normal
			triangle_type triangle = triangles_data[objIndex];
			vec3 e1 = triangle.vtx1.xyz - triangle.vtx0.xyz;
			vec3 e2 = triangle.vtx2.xyz - triangle.vtx0.xyz;
			vec3 normal = normalize(cross(e1,e2));
		
			finalPixelOut = shade(pointOnSurface, normal, triangle.colour.xyz);
			
			// COMPUTE FINAL PIXEL COLOUR USING SHADE() FUNCTION
		}
		else if (objType == OBB_TYPE)
		{
			// IMPLEMENT HERE
			// COMPUTE FINAL PIXEL COLOUR USING SHADE() FUNCTION
			//finalPixelOut = vec4(0.5f,0.5f,0.5f,1.0f);
			obb_type obb = obb_data[objIndex];
		
			vec3 c1 =  obb.centre.xyz + (obb.u_hu.xyz * obb.u_hu.w );
			vec3 c2 =  obb.centre.xyz - (obb.u_hu.xyz * obb.u_hu.w );

			vec3 c3 =  obb.centre.xyz + (obb.v_hv.xyz * obb.v_hv.w);
			vec3 c4 =  obb.centre.xyz - (obb.v_hv.xyz * obb.v_hv.w);

			vec3 c5 =  obb.centre.xyz + (obb.w_hw.xyz * obb.w_hw.w);
			vec3 c6 =  obb.centre.xyz - (obb.w_hw.xyz * obb.w_hw.w);

			float lowestDot = 1000000;
			vec3 potentialNormal;

			vec3[6] vecs = {c1,c2,c3,c4,c5,c6};

			for(int i = 0; i < 6; i++)
			{
				// Test on face 1
				float face1 = abs(dot(pointOnSurface - vecs[i], obb.u_hu.xyz));
				if(min(lowestDot, face1) == face1){
					lowestDot = face1;
					potentialNormal = (vecs[i] - obb.centre.xyz);
				}

				// Test on face 2
				float face3 = abs(dot(pointOnSurface - vecs[i], obb.v_hv.xyz));
				if(min(lowestDot, face3) == face3){
					lowestDot = face3;
					potentialNormal = (vecs[i] - obb.centre.xyz);
				}

				// Test on face 3
				float face5 = abs(dot(pointOnSurface - vecs[i], obb.w_hw.xyz));
				if(min(lowestDot, face5) == face5){
					lowestDot = face5;
					potentialNormal = (vecs[i] - obb.centre.xyz);
				}

			}

			finalPixelOut = shade(pointOnSurface, normalize(potentialNormal), obb.colour.xyz);
		}
	}
	return finalPixelOut;
}

layout (local_size_x = 20, local_size_y = 20) in;
void main()
{
	// pixel coordinate, x in [0,width-1], y in [0,height-1]
	ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
	vec4 finalPixel = castRay(pos);
	imageStore(outTexture, pos, finalPixel);
	return;
};

void sceneTest(in vec4 ray_dir,inout float lastT, inout int objIndex, inout int objType)
{

	// use camPos to know where is the ray origin.
	// example:
	vec3 rayOrigin = camPos;

	// test all spheres (in geomCount.x)
	for (int i=0;i< geomCount.x;i++)
	{
		sphere_type sphere = spheres_data[i];
		// IMPLEMENT HERE.
		// TEST SPHERE, UPDATE lastT, objIndex and objType if necessary.
		float minT = sphereTest(ray_dir.xyz, rayOrigin, sphere.centre_radius);
		if(minT != -1){
			if(min(lastT, minT) == minT || lastT == -1){
				lastT = minT;
				objIndex = i;
				objType = SPHERE_TYPE;
			}
		}
	}
	// lights (we render lights as spheres)
	for (int i=0;i<lightCount;i++)
	{
		light_type light = lights_data[i];
		// IMPLEMENT HERE.
		// TEST SPHERE (LIGHT), UPDATE lastT, objIndex and objType if necessary.
		float minT = sphereTest(ray_dir.xyz, rayOrigin, light.position);
		if(minT != -1){
			if(min(lastT, minT) == minT  || lastT == -1){
			lastT = minT;
			objIndex = i;
			objType = LIGHT_TYPE;
			}
		}
	}
	// planes
	for (int i = 0; i<geomCount.y; i++)
	{
		plane_type plane = planes_data[i];
		// IMPLEMENT HERE.
		// TEST PLANE, UPDATE lastT, objIndex and objType if necessary.
		float minT = planeTest(ray_dir.xyz,rayOrigin, plane.normal_d);

		if(minT != -1){
			if(min(lastT, minT) == minT  || lastT == -1){
			lastT = minT;
			objIndex = i;
			objType = PLANE_TYPE;
			}
		}
	}
	// triangles
	for (int i = 0; i<geomCount.z; i++)
	{
		triangle_type triangle = triangles_data[i];
		// IMPLEMENT HERE.
		// TEST TRIANGLE, UPDATE lastT, objIndex and objType if necessary.
		float minT = triangleTest(ray_dir.xyz,rayOrigin, triangle);

		if(minT != -1){
			if(min(lastT, minT) == minT  || lastT == -1){
			lastT = minT;
			objIndex = i;
			objType = TRIANGLE_TYPE;
			}
		}

	}
	// OBBs
	for (int i = 0; i < geomCount.w; i++)
	{
		obb_type obb = obb_data[i];
		// IMPLEMENT HERE.
		// TEST OBB, UPDATE lastT, objIndex and objType if necessary.
		float minT = obbTest(ray_dir.xyz,rayOrigin, obb);

		if(minT != -1){
			if(min(lastT, minT) == minT  || lastT == -1){
			lastT = minT;
			objIndex = i;
			objType = OBB_TYPE;
			}
		}

	}
}

float sphereTest(in vec3 rayDir, in vec3 rayOrigin, in vec4 centre_radius)
{		
		// Vec from origin to center of sphere
		vec3 oc = centre_radius.xyz - rayOrigin;
		vec3 rayDirNorm = normalize(rayDir);

		// Our t value that is going to be somewhere in the middle of the part of the ray that is in 
		// the sphere
		float t = dot(oc, rayDirNorm );
		
		// Get the point coordinates
		vec3 pointOnT = rayOrigin + t * rayDirNorm;

		// The length of the vector from the point to the center
		float y = length(centre_radius.xyz - pointOnT);
		
		// The circle can be defined as x^2 + y^2 = r^2

		// If y is bigger than radius we are going to end up with negative values when doing sqrt
		// Which means that there is no hit
		if( y > centre_radius.w)
			return -1;
		
		// Calculate the x so it can be applied to the t
		float x = sqrt(centre_radius.w * centre_radius.w - y * y);

		// Get the end points ( intersection points )
		float t1 = t - x;
		float t2 = t + x;

		// Check if they are valid
		if(t1 < 0.f)
			t1 = t2;

		if(t1 < 0.f)
			return -1;

		return t1;

}

float planeTest(vec3 rayDir, vec3 rayOrigin, vec4 plane_info)
{
	// IMPLEMENT HERE
	vec3 normal = normalize(plane_info.xyz);
	float dp = -(dot(normal, vec3(0,-plane_info.w,0)));
	
	// As long as ray direction and the normal of the plane is not perpendicular there will be
	// a intersection
	if(dot(normal,rayDir) == 0)
		return -1;
	
	float t = (-dp - (dot(normal, rayOrigin))) / (dot(normal,rayDir));
	if(t < 0)
		return -1;

	return t;
}

float triangleTest(vec3 rayDir, vec3 rayOrigin, triangle_type tri)
{
	// IMPLEMENT HERE
	vec3 e1 = (tri.vtx1.xyz - tri.vtx0.xyz);
	vec3 e2 = (tri.vtx2.xyz - tri.vtx0.xyz);
	vec3 s = (rayOrigin - tri.vtx0.xyz);

	mat3 m1;
	m1[0] = -rayDir;
	m1[1] = e1;
	m1[2] = e2;

	mat3 m2;
	m2[0] = s;
	m2[1] = e1; 
	m2[2] = e2;

	mat3 m3;
	m3[0] = -rayDir; 
	m3[1] = s; 
	m3[2] = e2;

	mat3 m4;
	m4[0] = -rayDir; 
	m4[1] = e1;
	m4[2] = s;

	float multiplier = 1.0/determinant(m1);

	float t = determinant(m2) * multiplier; 
	float u = determinant(m3) * multiplier; 
	float v = determinant(m4) * multiplier; 
	float w = 1.0 - u - v;

	if(w >= 0.0 && u > 0.0 && v > 0.0){
		if(u+v+w == 1.0){
			if(t > 0.0)
				return t;
		}
	}

	return -1;
}

float obbTest(vec3 rayDir, vec3 rayOrigin, obb_type o)
{
	// IMPLEMENT HERE
	float tmin = -1000000;
	float tmax = 1000000;
	// Distance from the start of the ray and the center of the box
	vec3 length = o.centre.xyz - rayOrigin;
	vec4 oobAxis;
	for(int i = 0; i < 3; i++){
		if(i == 0) oobAxis = o.u_hu;
		else if(i == 1) oobAxis = o.v_hv;
		else if(i == 2) oobAxis = o.w_hw;

	float e = dot(oobAxis.xyz, length);
	float f = dot(oobAxis.xyz, rayDir);

	if(abs(f) > 0.00000001f){
		float t1 = (e + oobAxis.w) / f;
		float t2 = (e - oobAxis.w) / f;

		// swap
		if(t1 > t2){
			float temp = t1;
			t1 = t2;
			t2 = temp;
		}

		if(t1 > tmin) tmin = t1;
		if(t2 < tmax) tmax = t2;

		if(tmin > tmax) 
			return -1;

		if(tmax < 0)
			return -1;

		

	}else if(-e - oobAxis.w > 0 || -e + oobAxis.w < 0 )
		return -1;
	}
	
	if(tmin > 0)
		return tmin;
	else
		return tmax;

	return -1;
};

// everythin in World Space
vec4 shade(in vec3 pointOnSurface, in vec3 normal, in vec3 colour)
{
	// ambient (combine with material color!)
	// 0.2 is arbitrary, that is the value used in the model solution.
	vec4 final_colour = vec4(0.2,0.2,0.2,1) * vec4(colour, 1.0f);
	float intensity = 15.0f;

	// diffuse, no attenuation.
	for (int i = 0; i < lightCount; i++)
	{
		vec4 light_pos = lights_data[i].position;
		vec4 light_colour = lights_data[i].colour;
		// IMPLEMENT HERE DIFFUSE SHADING

		vec3 vecToLight = light_pos.xyz - pointOnSurface;
		float distToLight = sqrt(dot(vecToLight, vecToLight));
		float cosVal  = dot(normalize(vecToLight),normal);
		
		float factor = 0.f;
		if(cosVal > 0.0f)
			factor = cosVal;

		vec3 color = colour * light_colour.xyz * factor * intensity * (1.0/distToLight);

		final_colour += vec4(color,1.0f);

	}
	// UPDATE THIS LINE TO ACCOUNT FOR SATURATION (PIXEL COLOUR CANNOT GO OVER 1.0)
	final_colour = min(final_colour, vec4(1.0, 1.0, 1.0, 1.0));
	return final_colour;
}

