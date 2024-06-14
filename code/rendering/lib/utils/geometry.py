import glob
import os
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import scipy.sparse as sp
import collections


def rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    return quat2mat(quat)


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def inv_4x4(mats):
    """Calculate the inverse of homogeneous transformations
    :param mats: [B, 4, 4]
    :return:
    """
    Rs = mats[:, :3, :3]
    ts = mats[:, :3, 3:]
    # R_invs = torch.transpose(Rs, 1, 2)
    R_invs = torch.inverse(Rs)
    t_invs = -torch.matmul(R_invs, ts)
    Rt_invs = torch.cat([R_invs, t_invs], dim=-1)   # [B, 3, 4]

    device = R_invs.device
    pad_row = torch.FloatTensor([0, 0, 0, 1]).to(device).view(1, 1, 4).expand(Rs.shape[0], -1, -1)
    mat_invs = torch.cat([Rt_invs, pad_row], dim=1)
    return mat_invs


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


def get_edge_unique(faces):
    """
    Parameters
    ------------
    faces: n x 3 int array
      Should be from a watertight mesh without degenerated triangles and intersection
    """
    faces = np.asanyarray(faces)

    # each face has three edges
    edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2))
    flags = edges[:, 0] < edges[:, 1]
    edges = edges[flags]
    return edges


def get_neighbors(edges):
    neighbors = collections.defaultdict(set)
    [(neighbors[edge[0]].add(edge[1]),
      neighbors[edge[1]].add(edge[0]))
     for edge in edges]

    max_index = edges.max() + 1
    array = [list(neighbors[i]) for i in range(max_index)]

    return array


def construct_degree_matrix(vnum, faces):
    row = col = list(range(vnum))
    value = [0] * vnum
    es = get_edge_unique(faces)
    for e in es:
        if e[0] < e[1]:
            value[e[0]] += 1
            value[e[0]] += 1

    dm = sp.coo_matrix((value, (row, col)), shape=(vnum, vnum), dtype=np.float32)
    return dm


def construct_neighborhood_matrix(vnum, faces):
    row = list()
    col = list()
    value = list()
    es = get_edge_unique(faces)
    for e in es:
        if e[0] < e[1]:
            row.append(e[0])
            col.append(e[1])
            value.append(1)
            row.append(e[1])
            col.append(e[0])
            value.append(1)

    nm = sp.coo_matrix((value, (row, col)), shape=(vnum, vnum), dtype=np.float32)
    return nm


def construct_laplacian_matrix(vnum, faces, normalized=False):
    edges = get_edge_unique(faces)
    neighbors = get_neighbors(edges)

    col = np.concatenate(neighbors)
    row = np.concatenate([[i] * len(n)
                          for i, n in enumerate(neighbors)])
    col = np.concatenate([col, np.arange(0, vnum)])
    row = np.concatenate([row, np.arange(0, vnum)])

    if normalized:
        data = [[1.0 / len(n)] * len(n) for n in neighbors]
        data += [[-1.0] * vnum]
    else:
        data = [[1.0] * len(n) for n in neighbors]
        data += [[-len(n) for n in neighbors]]

    data = np.concatenate(data)
    # create the sparse matrix
    matrix = sp.coo_matrix((data, (row, col)), shape=[vnum] * 2)
    return matrix


def rotationx_4x4(theta):
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, np.cos(theta / 180 * np.pi), np.sin(theta / 180 * np.pi), 0.0],
        [0.0, -np.sin(theta / 180 * np.pi), np.cos(theta / 180 * np.pi), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])


def rotationy_4x4(theta):
    return np.array([
        [np.cos(theta / 180 * np.pi), 0.0, np.sin(theta / 180 * np.pi), 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-np.sin(theta / 180 * np.pi), 0.0, np.cos(theta / 180 * np.pi), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])


def rotationz_4x4(theta):
    return np.array([
        [np.cos(theta / 180 * np.pi), np.sin(theta / 180 * np.pi), 0.0, 0.0],
        [-np.sin(theta / 180 * np.pi), np.cos(theta / 180 * np.pi), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])


def rotationx_3x3(theta):
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(theta / 180 * np.pi), np.sin(theta / 180 * np.pi)],
        [0.0, -np.sin(theta / 180 * np.pi), np.cos(theta / 180 * np.pi)],
    ])


def rotationy_3x3(theta):
    return np.array([
        [np.cos(theta / 180 * np.pi), 0.0, np.sin(theta / 180 * np.pi)],
        [0.0, 1.0, 0.0],
        [-np.sin(theta / 180 * np.pi), 0.0, np.cos(theta / 180 * np.pi)],
    ])


def rotationz_3x3(theta):
    return np.array([
        [np.cos(theta / 180 * np.pi), np.sin(theta / 180 * np.pi), 0.0],
        [-np.sin(theta / 180 * np.pi), np.cos(theta / 180 * np.pi), 0.0],
        [0.0, 0.0, 1.0],
    ])


def generate_point_grids(vol_res):
    x_coords = np.array(range(0, vol_res), dtype=np.float32)
    y_coords = np.array(range(0, vol_res), dtype=np.float32)
    z_coords = np.array(range(0, vol_res), dtype=np.float32)

    yv, xv, zv = np.meshgrid(x_coords, y_coords, z_coords)
    xv = np.reshape(xv, (-1, 1))
    yv = np.reshape(yv, (-1, 1))
    zv = np.reshape(zv, (-1, 1))
    pts = np.concatenate([xv, yv, zv], axis=-1)
    pts = pts.astype(np.float32)
    return pts


def infer_occupancy_value_grid_octree(test_res, pts, query_fn, init_res=64, ignore_thres=0.05):
    pts = np.reshape(pts, (test_res, test_res, test_res, 3))

    pts_ov = np.zeros([test_res, test_res, test_res])
    dirty = np.ones_like(pts_ov, dtype=np.bool)
    grid_mask = np.zeros_like(pts_ov, dtype=np.bool)

    reso = test_res // init_res
    while reso > 0:
        grid_mask[0:test_res:reso, 0:test_res:reso, 0:test_res:reso] = True
        test_mask = np.logical_and(grid_mask, dirty)

        pts_ = pts[test_mask]
        pts_ov[test_mask] = np.reshape(query_fn(pts_), pts_ov[test_mask].shape)

        if reso <= 1:
            break
        for x in range(0, test_res - reso, reso):
            for y in range(0, test_res - reso, reso):
                for z in range(0, test_res - reso, reso):
                    # if center marked, return
                    if not dirty[x + reso // 2, y + reso // 2, z + reso // 2]:
                        continue
                    v0 = pts_ov[x, y, z]
                    v1 = pts_ov[x, y, z + reso]
                    v2 = pts_ov[x, y + reso, z]
                    v3 = pts_ov[x, y + reso, z + reso]
                    v4 = pts_ov[x + reso, y, z]
                    v5 = pts_ov[x + reso, y, z + reso]
                    v6 = pts_ov[x + reso, y + reso, z]
                    v7 = pts_ov[x + reso, y + reso, z + reso]
                    v = np.array([v0, v1, v2, v3, v4, v5, v6, v7])
                    v_min = np.min(v)
                    v_max = np.max(v)
                    # this cell is all the same
                    if (v_max - v_min) < ignore_thres:
                        pts_ov[x:x + reso, y:y + reso, z:z + reso] = (v_max + v_min) / 2
                        dirty[x:x + reso, y:y + reso, z:z + reso] = False
        reso //= 2
    return pts_ov


def batch_rod2quat(rot_vecs):
    batch_size = rot_vecs.shape[0]

    angle = torch.norm(rot_vecs + 1e-16, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.cos(angle / 2)
    sin = torch.sin(angle / 2)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)

    qx = rx * sin
    qy = ry * sin
    qz = rz * sin
    qw = cos-1.0

    return torch.cat([qx,qy,qz,qw], dim=1)


def batch_quat2matrix(rvec):
    '''
    args:
        rvec: (B, N, 4)
    '''
    B, N, _ = rvec.size()

    theta = torch.sqrt(1e-5 + torch.sum(rvec ** 2, dim=2))
    rvec = rvec / theta[:, :, None]
    return torch.stack((
        1. - 2. * rvec[:, :, 1] ** 2 - 2. * rvec[:, :, 2] ** 2,
        2. * (rvec[:, :, 0] * rvec[:, :, 1] - rvec[:, :, 2] * rvec[:, :, 3]),
        2. * (rvec[:, :, 0] * rvec[:, :, 2] + rvec[:, :, 1] * rvec[:, :, 3]),

        2. * (rvec[:, :, 0] * rvec[:, :, 1] + rvec[:, :, 2] * rvec[:, :, 3]),
        1. - 2. * rvec[:, :, 0] ** 2 - 2. * rvec[:, :, 2] ** 2,
        2. * (rvec[:, :, 1] * rvec[:, :, 2] - rvec[:, :, 0] * rvec[:, :, 3]),

        2. * (rvec[:, :, 0] * rvec[:, :, 2] - rvec[:, :, 1] * rvec[:, :, 3]),
        2. * (rvec[:, :, 0] * rvec[:, :, 3] + rvec[:, :, 1] * rvec[:, :, 2]),
        1. - 2. * rvec[:, :, 0] ** 2 - 2. * rvec[:, :, 1] ** 2
        ), dim=2).view(B, N, 3, 3)


def get_posemap(map_type, n_joints, parents, n_traverse=1, normalize=True):
    pose_map = torch.zeros(n_joints,n_joints-1)
    if map_type == 'parent':
        for i in range(n_joints-1):
            pose_map[i+1,i] = 1.0
    elif map_type == 'children':
        for i in range(n_joints-1):
            parent = parents[i+1]
            for j in range(n_traverse):
                pose_map[parent, i] += 1.0
                if parent == 0:
                    break
                parent = parents[parent]
        if normalize:
            pose_map /= pose_map.sum(0,keepdim=True)+1e-16
    elif map_type == 'both':
        for i in range(n_joints-1):
            pose_map[i+1,i] += 1.0
            parent = parents[i+1]
            for j in range(n_traverse):
                pose_map[parent, i] += 1.0
                if parent == 0:
                    break
                parent = parents[parent]
        if normalize:
            pose_map /= pose_map.sum(0,keepdim=True)+1e-16
    else:
        raise NotImplementedError('unsupported pose map type [%s]' % map_type)
    pose_map = torch.cat([torch.zeros(n_joints, 1), pose_map], dim=1)
    return pose_map


def vertices_to_triangles(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3)
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]


def calc_face_normals(vertices, faces):
    assert len(vertices.shape) == 3
    assert len(faces.shape) == 2
    if isinstance(faces, np.ndarray):
        faces = torch.from_numpy(faces.astype(np.int64)).to(vertices.device)

    batch_size, pt_num = vertices.shape[:2]
    face_num = faces.shape[0]

    triangles = vertices_to_triangles(vertices, faces.unsqueeze(0).expand(batch_size, -1, -1))
    triangles = triangles.reshape((batch_size * face_num, 3, 3))
    v10 = triangles[:, 0] - triangles[:, 1]
    v12 = triangles[:, 2] - triangles[:, 1]
    # pytorch normalize divides by max(norm, eps) instead of (norm+eps) in chainer
    normals = F.normalize(torch.cross(v10, v12), eps=1e-5)
    normals = normals.reshape((batch_size, face_num, 3))

    return normals


def calc_vert_normals_numpy(vertices, faces):
    assert len(vertices.shape) == 2
    assert len(faces.shape) == 2

    nmls = np.zeros_like(vertices)
    fv0 = vertices[faces[:, 0]]
    fv1 = vertices[faces[:, 1]]
    fv2 = vertices[faces[:, 2]]
    face_nmls = np.cross(fv1-fv0, fv2-fv0, axis=-1)
    face_nmls = face_nmls / (np.linalg.norm(face_nmls, axis=-1, keepdims=True) + 1e-20)
    for f, fn in zip(faces, face_nmls):
        nmls[f] += fn
    nmls = nmls / (np.linalg.norm(nmls, axis=-1, keepdims=True) + 1e-20)
    return nmls


def glUV2torchUV(gl_uv):
    torch_uv = torch.stack([
        gl_uv[..., 0]*2.0-1.0,
        gl_uv[..., 1]*-2.0+1.0
    ], dim=-1)
    return torch_uv


def normalize_vert_bbox(verts, dim=-1, per_axis=False):
    bbox_min = torch.min(verts, dim=dim, keepdim=True)[0]
    bbox_max = torch.max(verts, dim=dim, keepdim=True)[0]
    verts = verts - 0.5 * (bbox_max + bbox_min)
    if per_axis:
        verts = 2 * verts / (bbox_max - bbox_min)
    else:
        verts = 2 * verts / torch.max(bbox_max-bbox_min, dim=dim, keepdim=True)[0]
    return verts
