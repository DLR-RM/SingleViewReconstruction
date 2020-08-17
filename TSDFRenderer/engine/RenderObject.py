from __future__ import with_statement
from skimage import measure
from OpenGL.GL import *
from OpenGL.GLUT import *
import numpy as np

main_path = os.path.join(os.path.dirname(__file__), "..")
single_view_path = os.path.abspath(os.path.join(main_path, "..", "SingleViewReconstruction"))
import sys
sys.path.append(main_path)
sys.path.append(single_view_path)

from src.utils import StopWatch
from engine.math_gl import *
from threading import Lock


def opengl_error_check():
    error = glGetError()
    if error != GL_NO_ERROR:
        print("OPENGL_ERROR: ", gluErrorString(error))

class RenderObject(object):

    def __init__(self, name):
        self._name = name

    def render(self, vp, mapping):
        print("This fct. should be overwritten")

    def delete(self):
        print("This fct. should be overwritten")

    def flip_collapse_mode(self):
        pass


class PlaneObject(RenderObject):

    def __init__(self, name, image_id=None, move_vec=None):
        self._name = name
        if move_vec is None:
            move_vec = np.array([0,0,0])
        self._plane_verts = np.array([np.array(ele) for ele in [[0,0,0], [1,0,0], [1,1,0], [0,1,0]]]).astype(GLfloat)
        self._plane_normals = np.array([[0,0,1]] * 4).astype(GLfloat)
        self._plane_indices = np.array([0,1,2,0,2,3]).astype(GLuint)
        self.move_vec(move_vec)

    def render(self, vp, mapping):
        print("This fct. should be overwritten")

    def delete(self):
        print("This fct. should be overwritten")

    def flip_collapse_mode(self):
        pass

    def render(self, vp, mapping):
        if not self._inited:
            self.init()
        null = c_void_p(0)
        mvp = vp * self._model_matrix

        unproj_mat = mat4.identity()
        glUniformMatrix4fv(mapping['MVP'], 1, GL_FALSE,mvp.data)
        glUniformMatrix4fv(mapping['unproj_mat'], 1, GL_FALSE, unproj_mat.data)
        glUniformMatrix4fv(mapping['M'], 1, GL_FALSE, self._model_matrix.data)

        # activate the texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._mapping['texture'])
        # Set our "myTextureSampler" sampler to user Texture Unit 0
        glUniform1i(mapping['texture_id'], 0)

        glUniform1i(mapping['UseTexture'], self._used_texture_value)

        # activate vertex data
        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, self._mapping['vertex_buffer'])
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, null)

        # activate normal data
        glEnableVertexAttribArray(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._mapping['normal_buffer'])
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, null)

        # activate indices
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._mapping['elementbuffer'])

        # Draw the triangles
        glDrawElements(GL_TRIANGLES, len(self._indices), GL_UNSIGNED_INT,	null)

        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)

    def _bind_to_card(self, image_id):
        with TSDFRenderObject.bind_to_card_lock:
            self._mapping['texture'] = image_id
            self._mapping['vertex_buffer'] = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self._mapping['vertex_buffer'])
            glBufferData(GL_ARRAY_BUFFER, len(self._plane_verts) * 4 * 3, self._plane_verts, GL_STATIC_DRAW)

            self._mapping['normal_buffer'] = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self._mapping['normal_buffer'])
            glBufferData(GL_ARRAY_BUFFER, len(self._plane_normals) * 4 * 3, self._plane_normals, GL_STATIC_DRAW)

            # Generate a buffer for the indices as well
            self._mapping['elementbuffer'] = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._mapping['elementbuffer'])
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(self._plane_indices) * 4, self._plane_indices, GL_STATIC_DRAW)

    def move_vec(self, move_vec):
        self._model_matrix = mat4.identity()
        self._model_matrix[0][0] = 0
        self._model_matrix[1][0] = 1
        self._model_matrix[0][1] = -1
        self._model_matrix[1][1] = 0
        if move_vec is not None:
            trans_mat = mat4.identity()
            trans_mat.translate(vec3(move_vec[0], move_vec[1], move_vec[2]))
            self._model_matrix = trans_mat * self._model_matrix


class TSDFRenderObject(RenderObject):
    bind_to_card_lock = Lock()

    def __init__(self, voxel, image=None, name=None, move_vec=None, use_preloaded_img=None, marching_cubes_result=None, voxel_size=None):
        if name:
            super(TSDFRenderObject, self).__init__(name)
        else:
            super(TSDFRenderObject, self).__init__('TSDF_Volume')
        self._use_texture = True
        self._flip_plane_mode = False
        if image is not None:
            self._used_texture_value = 0 if np.mean(np.var(image)) < 1e-3 else 1
        else:
            self._used_texture_value = 0

        if marching_cubes_result is None and voxel is not None:
            # voxel set up
            verts, faces, normals, _ = TSDFRenderObject.perform_marching_cubes(voxel)
            print("Verts: {}, faces: {}".format(len(verts), len(faces)))
            voxel_size = voxel.shape[0]
        else:
            verts, faces, normals, _ = marching_cubes_result


        verts = np.array(verts).astype(GLfloat) / float(voxel_size)

        normals = np.array(normals).astype(GLfloat)
        faces = np.array(faces).flatten()

        self._indexed_vertices = verts[faces]
        self._indexed_normals = normals[faces]
        self._indices = np.arange(self._indexed_vertices.shape[0], dtype=GLuint)
        self.use_unproject = True

        if use_preloaded_img is None:
            # image set up
            if image is not None and len(image.shape) == 3 and image.shape[0] == 512 and image.shape[1] == 512 and image.shape[2] == 4:
                self._image = image.astype(np.uint8)
            else:
                if image is not None:
                    if len(image.shape) == 3 and image.shape[0] == 512 and image.shape[1] == 512 and image.shape[2] == 3:
                        self._image = np.concatenate([image.astype(np.uint8), np.ones((512,512,1), dtype=np.uint8) * 255], axis=2).astype(np.uint8)
                    else:
                        print("The image has not the right form: " + str(image.shape) + ", for " + str(self._name))
                        self._image = np.zeros((512, 512, 4), dtype=np.uint8)
                else:
                    self._image = np.zeros((512, 512, 4), dtype=np.uint8)
        self._use_preloaded_img = use_preloaded_img

        self._mapping = {}
        self.move_vec(move_vec)
        self._inited = False

    @staticmethod
    def perform_marching_cubes(voxel):
        threshold = np.max([np.min(voxel) + 1e-8, 0])
        print('Use threshold: {}, min: {}, max: {}'.format(threshold, np.min(voxel), np.max(voxel)))
        return measure.marching_cubes(voxel, threshold)

    def init(self, use_preloaded_img_id=None):
        if use_preloaded_img_id is not None:
            self._use_preloaded_img = use_preloaded_img_id
        self._bind_to_card()
        self._inited = True

    def flip_color_mode(self):
        self._used_texture_value = int((self._used_texture_value + 1) % 2)

    def flip_plane_mode(self):
        self._flip_plane_mode = not self._flip_plane_mode

    def flip_unproject(self):
        self.use_unproject = not self.use_unproject


    def _bind_to_card(self):
        with TSDFRenderObject.bind_to_card_lock:
            if self._use_preloaded_img is None:
                self._mapping['texture'] = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, self._mapping['texture'])
                glPixelStorei(GL_UNPACK_ALIGNMENT,1)
                glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
                glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
                glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

                glTexImage2D(GL_TEXTURE_2D, 0, 3, 512, 512, 0,
                             GL_RGBA, GL_UNSIGNED_BYTE, self._image)
            else:
                self._mapping['texture'] = self._use_preloaded_img
            opengl_error_check()

            self._mapping['vertex_buffer'] = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self._mapping['vertex_buffer'])
            glBufferData(GL_ARRAY_BUFFER, len(self._indexed_vertices) * 4 * 3, self._indexed_vertices, GL_STATIC_DRAW)

            self._mapping['normal_buffer'] = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self._mapping['normal_buffer'])
            glBufferData(GL_ARRAY_BUFFER, len(self._indexed_normals) * 4 * 3, self._indexed_normals, GL_STATIC_DRAW)

            # Generate a buffer for the indices as well
            self._mapping['elementbuffer'] = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._mapping['elementbuffer'])
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(self._indices) * 4, self._indices, GL_STATIC_DRAW)



            # glBufferData(GL_ARRAY_BUFFER, len(self._plane_verts) * 4 * 3, self._plane_verts, GL_STATIC_DRAW)
            # glBufferData(GL_ARRAY_BUFFER, len(self._plane_normals) * 4 * 3, self._plane_normals, GL_STATIC_DRAW)
            # glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(self._plane_indices) * 4, self._plane_indices, GL_STATIC_DRAW)

    def render(self, vp, mapping):
        if not self._inited:
            self.init()
        null = c_void_p(0)
        mvp = vp * self._model_matrix
        self.unproj_mat = mat4.identity()
        self._used_unproject_value = 1 if self.use_unproject else 0
        if self.use_unproject:
            scale = 1.0
            xFov = 0.5
            near = 1.0
            far = 4.0
            yFov = 0.388863
            width = 1. / np.tan(xFov)
            height = 1. / np.tan(yFov)
            proj_mat = np.array([[width/scale, 0, 0, 0],[0, height/scale, 0, 0],[0, 0, (near + far) / (near - far)/scale, (2*near+far) / (near- far)/scale], [0, 0, 1, 0]])
            result = np.linalg.inv(proj_mat)
            result = result.astype(GLfloat)
            for i in range(4):
                for j in range(4):
                    if abs(result[i,j]) < 1e-5:
                        self.unproj_mat[i][j] = 0
                    else:
                        self.unproj_mat[i][j] = result[i,j]

            # print(unproj_mat)
        glUniformMatrix4fv(mapping['MVP'], 1, GL_FALSE,mvp.data)
        glUniformMatrix4fv(mapping['unproj_mat'], 1, GL_TRUE, self.unproj_mat.data)
        glUniform1i(mapping['use_unproject'], self._used_unproject_value)
        glUniformMatrix4fv(mapping['M'], 1, GL_FALSE, self._model_matrix.data)

        if self._used_texture_value == 1:
            # activate the texture
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self._mapping['texture'])
            # Set our "myTextureSampler" sampler to user Texture Unit 0
            glUniform1i(mapping['texture_id'], 0)

        glUniform1i(mapping['UseTexture'], self._used_texture_value)

        # activate vertex data
        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, self._mapping['vertex_buffer'])
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, null)

        # activate normal data
        glEnableVertexAttribArray(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._mapping['normal_buffer'])
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, null)

        # activate indices
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._mapping['elementbuffer'])

        # Draw the triangles
        glDrawElements(GL_TRIANGLES, len(self._indices), GL_UNSIGNED_INT,	null)

        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)

    def delete(self):
        glDeleteBuffers(1, [self._mapping['vertex_buffer']])
        glDeleteBuffers(1, [self._mapping['normal_buffer']])
        glDeleteBuffers(1, [self._mapping['elementbuffer']])
        glDeleteBuffers(1, [self._mapping['texture']])

    def get_texture_id(self):
        if 'texture' in self._mapping:
            return self._mapping['texture']
        else:
            # this has not been set
            return 0

    def move_vec(self, move_vec):
        self._model_matrix = mat4.identity()
        self._model_matrix[0][0] = 0
        self._model_matrix[1][0] = 1
        self._model_matrix[0][1] = -1
        self._model_matrix[1][1] = 0
        if move_vec is not None:
            trans_mat = mat4.identity()
            trans_mat.translate(vec3(move_vec[0], move_vec[1], move_vec[2]))
            self._model_matrix = trans_mat * self._model_matrix


class TSDFRenderObjectTrueAndPrediction(RenderObject):

    def __init__(self, prediction, true, image=None, name=None, move_vec=None, diff_vec=None, marching_cube_res_prediction=None, marching_cube_res_true=None, voxel_size=None):
        if name is None:
            name = 'TSDF_object'
        self._true_object = TSDFRenderObject(true, image, name + '_true', move_vec, marching_cubes_result=marching_cube_res_true, voxel_size=voxel_size)
        if move_vec is None:
            move_vec = np.zeros(3)
        if diff_vec is None:
            diff_vec = np.array([0,1.2,0])
        texture_id = self._true_object.get_texture_id() # if this returns 0 the texture id has to be set later
        self._prediction_move_vec = np.array(move_vec) + diff_vec
        self._prediction_object = TSDFRenderObject(prediction, image, name + '_true', self._prediction_move_vec, texture_id, marching_cube_res_prediction, voxel_size)
        self._true_move_vec = move_vec
        self._separated = True
        self._inited = False

    def init(self):
        if not self._inited:
            self._true_object.init()
            self._prediction_object.init(self._true_object.get_texture_id())
            self._inited = True

    def render(self, vp, mapping):
        self.init()
        self._true_object.render(vp, mapping)
        self._prediction_object.render(vp, mapping)

    def delete(self):
        self._true_object.delete()
        self._prediction_object.delete()

    def flip_collapse_mode(self):
        self._separated = False if self._separated else True
        if self._separated:
            self._prediction_object.move_vec(self._prediction_move_vec)
        else:
            self._prediction_object.move_vec(self._true_move_vec)

    def flip_color_mode(self):
        self._prediction_object.flip_color_mode()
        self._true_object.flip_color_mode()

    def flip_plane_mode(self):
        self._prediction_object.flip_plane_mode()
        self._true_object.flip_plane_mode()

    def flip_unproject(self):
        self._prediction_object.flip_unproject()
        self._true_object.flip_unproject()


