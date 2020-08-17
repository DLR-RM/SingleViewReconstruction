#version 410 core

// Interpolated values from the vertex shaders
in vec2 UV;
in vec3 Position_worldspace;
in vec3 Position_modelspace;
in vec3 Normal_modelspace;
in vec3 Normal_cameraspace;
in vec3 EyeDirection_cameraspace;
in vec3 LightDirection_cameraspace;

// Ouput data
out vec3 color;

// Values that stay constant for the whole mesh.
uniform sampler2D myTextureSampler;
uniform mat4 MV;
uniform vec3 LightPosition_worldspace;
uniform int UseTexture; // is 1 if texture is used 0 if white color should be used


void main(){

	// Material properties
	if(UseTexture == 0){
        color = ((Normal_modelspace * 0.5) + vec3(0.5, 0.5, 0.5));
    }else{
        color = texture2D( myTextureSampler, UV ).rgb;
    }
}

