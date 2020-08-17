from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GLUT.special import *
from OpenGL.GL.shaders import *

def LoadShaders(vertex_file_path, fragment_file_path):
	# Create the shaders
	VertexShaderID = glCreateShader(GL_VERTEX_SHADER)
	FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER)

	# Read the Vertex Shader code from the file
	VertexShaderCode = ""
	with open(vertex_file_path, 'r') as fr:
		for line in fr:
			VertexShaderCode += line
		# alternatively you could use fr.readlines() and then join in to a single string

	FragmentShaderCode = ""
	with open(fragment_file_path, 'r') as fr:
		for line in fr:
			FragmentShaderCode += line
		# alternatively you could use fr.readlines() and then join in to a single string

	# Compile Vertex Shader
	print("Compiling shader: %s"%(vertex_file_path))
	glShaderSource(VertexShaderID, VertexShaderCode)
	glCompileShader(VertexShaderID)

	# Check Vertex Shader
	result = glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS)
	if not result:
		raise RuntimeError(glGetShaderInfoLog(VertexShaderID))

	# Compile Fragment Shader
	print("Compiling shader: %s"%(fragment_file_path))
	glShaderSource(FragmentShaderID,FragmentShaderCode)
	glCompileShader(FragmentShaderID)

	# Check Fragment Shader
	result = glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS)
	if not result:
		raise RuntimeError(glGetShaderInfoLog(FragmentShaderID))



	# Link the program
	print("Linking program")
	ProgramID = glCreateProgram()
	glAttachShader(ProgramID, VertexShaderID)
	glAttachShader(ProgramID, FragmentShaderID)
	glLinkProgram(ProgramID)

	# Check the program
	result = glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS)
	if not result:
		raise RuntimeError(glGetShaderInfoLog(ProgramID))

	glDeleteShader(VertexShaderID)
	glDeleteShader(FragmentShaderID)

	return ProgramID