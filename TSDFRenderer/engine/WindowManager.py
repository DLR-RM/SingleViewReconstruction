import cyglfw3 as glfw

from OpenGL.GL import *
from OpenGL.GL.ARB import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GLUT.special import *
from OpenGL.GL.shaders import *
from engine.math_gl import *
import engine.common as common
import math as mathf


def windows_focus_callback(window, focus):
    print("focus")

class WindowManager(object):

    def __init__(self, window_size, title=None):
        self.window = None
        self._window_size = window_size
        if title:
            self._title = title
        else:
            self._title = 'TSDF renderer'
        self._mapping = {}
        self._render_objects = []
        self._org_camera_position = vec3( 0.5, -0.5, 3 )
        self._camera_position = self._org_camera_position.copy()
        self._org_angle = (3.14, 0.0)
        self._horizontalAngle = self._org_angle[0]
        self._verticalAngle = self._org_angle[1]
        self._focus = False
        self._lastTime = None
        self._last_key_pressed = glfw.GetTime()

    def init(self):
        assert glfw.Init(), 'Glfw Init failed!'
        self.window = glfw.CreateWindow(self._window_size[0], self._window_size[1], self._title, None)
        glfw.WindowHint(glfw.SAMPLES, 4)
        glfw.WindowHint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.WindowHint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.WindowHint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.WindowHint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        glfw.MakeContextCurrent(self.window)
        glfw.SetInputMode(self.window, glfw.STICKY_KEYS, True)

        # Enable depth test
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_3D)
        # Accept fragment if it closer to the camera than the former one
        glDepthFunc(GL_LESS)
        # disable vsync
        glfw.SwapInterval(0)
        self._init_shaders()

    def _init_shaders(self):
        self._mapping['vertex_array_id'] = glGenVertexArrays(1)
        glBindVertexArray(self._mapping['vertex_array_id'])
        vertex_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "VertexShader.glsl")
        fragment_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "FragmentShader.glsl")
        self._mapping['program_id'] = common.LoadShaders(vertex_path, fragment_path)
        program_id = self._mapping['program_id']
        self._mapping['texture_id'] = glGetUniformLocation(program_id, "myTextureSampler")
        use_obj = True
        if use_obj:
            self._mapping['obj_class_id'] = glGetUniformLocation(program_id, "objClassSampler")


        # Get a handle for our "MVP" uniform
        self._mapping['MVP'] = glGetUniformLocation(program_id, "MVP")
        self._mapping['unproj_mat'] = glGetUniformLocation(program_id, "unproj_mat")
        self._mapping['V'] = glGetUniformLocation(program_id, "V")
        self._mapping['M'] = glGetUniformLocation(program_id, "M")
        self._mapping['light_id'] = glGetUniformLocation(program_id, "LightPosition_worldspace")
        self._mapping['UseTexture'] = glGetUniformLocation(program_id, "UseTexture")
        self._mapping['use_unproject'] = glGetUniformLocation(program_id, "use_unproject")

    def _windows_focus_callback(self, focus):
        self._focus = focus

    def add_render_object(self, object):
        self._render_objects.append(object)

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT)
        glUseProgram(self._mapping['program_id'])
        vp = self._compute_matrices_from_input()
        lightPos = vec3(2,2,6)
        glUniform3f(self._mapping['light_id'], lightPos.x, lightPos.y, lightPos.z)
        for object in self._render_objects:
            object.render(vp, self._mapping)

        glfw.SwapBuffers(self.window)

        # Poll for and process events
        glfw.PollEvents()


    def run_window(self):
        last_time = glfw.GetTime()
        frames = 0
        while glfw.GetKey(self.window,glfw.KEY_ESCAPE) != glfw.PRESS and not glfw.WindowShouldClose(self.window):

            current_time = glfw.GetTime()
            if current_time - last_time >= 1.0:
                glfw.SetWindowTitle(self.window, "Loaded! Running at FPS: %d"%(frames))
                frames = 0
                last_time = current_time
            self.render()
            frames += 1

        for object in self._render_objects:
            object.delete()
        glDeleteProgram(self._mapping['program_id'])
        glDeleteVertexArrays(1, [self._mapping['vertex_array_id']])
        glfw.Terminate()

    def _is_key_pressed(self, key):
        if isinstance(key, list):
            for ele in key:
                if self._is_key_pressed(ele):
                    return True
            return False
        else:
            return glfw.GetKey( self.window, key) == glfw.PRESS

    def _compute_matrices_from_input(self):
        old_focus = self._focus
        self._focus = glfw.GetWindowAttrib(self.window, glfw.FOCUSED)
        if old_focus != self._focus and self._focus:
            glfw.SetCursorPos(self.window, self._window_size[0]/2, self._window_size[1]/2)

        FoV = 45
        mouse_speed =  0.001
        speed = 5.0
        # glfwGetTime is called only once, the first time this function is called
        if self._lastTime is None:
            self._lastTime = glfw.GetTime()

        currentTime = glfw.GetTime()
        if self._focus:
            if self._is_key_pressed(glfw.KEY_O):
                self._camera_position = self._org_camera_position.copy()
                self._horizontalAngle = self._org_angle[0]
                self._verticalAngle = self._org_angle[1]

            deltaTime = currentTime - self._lastTime
            #if deltaTime > 0.01:
            xpos,ypos = glfw.GetCursorPos(self.window)

            # Reset mouse position for next frame
            if xpos != self._window_size[0]/2 or ypos != self._window_size[1]/2:
                glfw.SetCursorPos(self.window, self._window_size[0]/2, self._window_size[1]/2)

            # Compute new orientation
            self._horizontalAngle += mouse_speed * float(self._window_size[0]/2.0 - xpos )
            self._verticalAngle   += mouse_speed * float( self._window_size[1]/2.0 - ypos )

        # Direction : Spherical coordinates to Cartesian coordinates conversion
        direction = vec3(
            mathf.cos(self._verticalAngle) * mathf.sin(self._horizontalAngle),
            mathf.sin(self._verticalAngle),
            mathf.cos(self._verticalAngle) * mathf.cos(self._horizontalAngle)
        )

        # Right vector
        right = vec3(
            mathf.sin(self._horizontalAngle - 3.14/2.0),
            0.0,
            mathf.cos(self._horizontalAngle - 3.14/2.0)
        )

        # Up vector
        up = vec3.cross( right, direction )

        if self._focus:
            # Move forward
            if self._is_key_pressed([glfw.KEY_W, glfw.KEY_UP]):
                self._camera_position += direction * deltaTime * speed

            # Move backward
            if self._is_key_pressed([glfw.KEY_S, glfw.KEY_DOWN]):
                self._camera_position -= direction * deltaTime * speed

            # Strafe right
            if self._is_key_pressed([glfw.KEY_D, glfw.KEY_RIGHT]):
                self._camera_position += right * deltaTime * speed

            # Strafe left
            if self._is_key_pressed([glfw.KEY_A, glfw.KEY_LEFT]):
                self._camera_position -= right * deltaTime * speed

            if self._is_key_pressed(glfw.KEY_C):
                new_press =  glfw.GetTime()
                if new_press - self._last_key_pressed > 0.15:
                    for object in self._render_objects:
                        object.flip_collapse_mode()
                    self._last_key_pressed = new_press

            if self._is_key_pressed(glfw.KEY_T):
                new_press =  glfw.GetTime()
                if new_press - self._last_key_pressed > 0.15:
                    for object in self._render_objects:
                        object.flip_color_mode()
                    self._last_key_pressed = new_press

            if self._is_key_pressed(glfw.KEY_U):
                new_press =  glfw.GetTime()
                if new_press - self._last_key_pressed > 0.15:
                    for object in self._render_objects:
                        object.flip_unproject()
                    self._last_key_pressed = new_press



        ProjectionMatrix = mat4.perspective(FoV, self._window_size[0] / float(self._window_size[1]), 0.1, 100.0)
        ViewMatrix       = mat4.lookat(self._camera_position, self._camera_position+direction, up)


        vp = ProjectionMatrix * ViewMatrix

        glUniformMatrix4fv(self._mapping['V'], 1, GL_FALSE, ViewMatrix.data)
        self._lastTime = currentTime

        return vp



