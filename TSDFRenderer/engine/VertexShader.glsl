#version 410 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec3 vertexNormal_modelspace;

// Output data ; will be interpolated for each fragment.
out vec2 UV;
out vec3 Position_modelspace;
out vec3 Position_worldspace;
out vec3 Normal_cameraspace;
out vec3 Normal_modelspace;
out vec3 EyeDirection_cameraspace;
out vec3 LightDirection_cameraspace;

// Values that stay constant for the whole mesh.
uniform mat4 MVP;
uniform mat4 V;
uniform mat4 M;
uniform mat4 unproj_mat;
uniform vec3 LightPosition_worldspace;
uniform int use_unproject;

void main(){


	// Output position of the vertex, in clip space : MVP * position
	if(use_unproject == 1){
        vec3 coord = vertexPosition_modelspace;
        for(int i = 0; i < 3; ++i){
            coord[i] *= 2.0;
        }
        for(int i = 0; i < 3; ++i){
           coord[i] -= 1.0;
        }

        float temp = coord[0];
        coord[0] = coord[1];
        coord[1] = temp;
        coord[0] *= -1.0;

        vec4 pre = vec4(coord, 1.0);
        vec4 result = unproj_mat * pre;
        float scale = result[3];
        float fac = 0.7;
        for(int i = 0; i < 4; ++i){
            result[i] /= scale;
            result[i] *= fac;
        }
        // rotate it around the z-axis
        vec4 res = vec4(-result[1]+0.5, result[0]+0.5, result[2]+2, 1.0);
        gl_Position = MVP * res;
	}else{
	    gl_Position = MVP * vec4(vertexPosition_modelspace,1);
	}

	// Position of the vertex, in worldspace : M * position
	Position_worldspace = (M * vec4(vertexPosition_modelspace,1)).xyz;

	// Vector that goes from the vertex to the camera, in camera space.
	// In camera space, the camera is at the origin (0,0,0).
	vec3 vertexPosition_cameraspace = ( V * M * vec4(vertexPosition_modelspace,1)).xyz;
	EyeDirection_cameraspace = vec3(0,0,0) - vertexPosition_cameraspace;

	// Vector that goes from the vertex to the light, in camera space. M is ommited because it's identity.
    vec4 light_pos_world_space = M * vec4(0.5, 0.5, 0.5, 1);
	vec3 LightPosition_cameraspace = ( V * light_pos_world_space).xyz;
	LightDirection_cameraspace = LightPosition_cameraspace + EyeDirection_cameraspace;

	// Normal of the the vertex, in camera space
	Normal_cameraspace = ( V * M * vec4(vertexNormal_modelspace,0)).xyz; // Only correct if ModelMatrix does not scale the model ! Use its inverse transpose if not.

	// UV of the vertex. No special space for this one.
	UV = vec2(vertexPosition_modelspace[1], vertexPosition_modelspace[0]);
    Position_modelspace = vertexPosition_modelspace;
    Normal_modelspace = vertexNormal_modelspace;
}