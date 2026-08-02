import cv2
import math
import numpy as np


def _as_aop_ellipse(element):
    return (element[0], (element[1][1], element[1][0]), element[2] - 90)


def _ellipse_left_right_points(ps_ellipse):
    element = _as_aop_ellipse(ps_ellipse)
    cx, cy = element[0]
    major = element[1][0]
    angle = element[2] * math.pi / 180.0
    p1 = np.array([
        cx - major / 2 * math.cos(angle),
        cy - major / 2 * math.sin(angle),
    ], dtype=np.float64)
    p2 = np.array([
        cx + major / 2 * math.cos(angle),
        cy + major / 2 * math.sin(angle),
    ], dtype=np.float64)
    if p1[0] <= p2[0]:
        return p1, p2
    return p2, p1


def _fh_tangent_point(fh_ellipse, anchor_point):
    element = _as_aop_ellipse(fh_ellipse)
    center = np.array(element[0], dtype=np.float64)
    a = element[1][0] / 2
    b = element[1][1] / 2
    angle = 2 * math.pi * element[2] / 360

    dp = np.array([[anchor_point[0] - center[0]], [anchor_point[1] - center[1]]], dtype=np.float64)
    trans_to_axis = np.array([
        [math.cos(-angle), -math.sin(-angle)],
        [math.sin(-angle), math.cos(-angle)],
    ])
    trans_to_image = np.array([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ])
    px, py = (trans_to_axis @ dp).flatten()

    if abs(px ** 2 - a ** 2) < 1e-6:
        px += 1e-3
    discriminant = b ** 2 * px ** 2 + a ** 2 * py ** 2 - a ** 2 * b ** 2
    if discriminant >= 0:
        slope = (px * py - math.sqrt(discriminant)) / (px ** 2 - a ** 2)
    else:
        slope = 0.0
    bias = py - slope * px
    denom = 2 * (1 / a ** 2 + slope ** 2 / b ** 2)
    if abs(denom) < 1e-6:
        return center.copy()
    tx = (-2 * slope * bias / b ** 2) / denom
    ty = tx * slope + bias
    tangent_axis = np.array([[tx], [ty]], dtype=np.float64)
    tangent = (trans_to_image @ tangent_axis).flatten() + center
    return tangent


def aop_geometry(fh_ellipse, ps_ellipse):
    """Return AoP and points used to draw it.

    Points are image-space ``(x, y)`` floats:
    - ``ps_left``: leftmost endpoint of the PS major axis
    - ``ps_right``: rightmost endpoint of the PS major axis, the AoP vertex
    - ``fh_contact``: tangent/contact point on FH from ``ps_right``
    """
    ps_left, ps_right = _ellipse_left_right_points(ps_ellipse)
    fh_contact = _fh_tangent_point(fh_ellipse, ps_right)

    v_ps = ps_left - ps_right
    v_fh = fh_contact - ps_right
    norm = np.linalg.norm(v_ps) * np.linalg.norm(v_fh)
    if norm <= 1e-6:
        raise ValueError('Degenerate AoP line geometry')
    cos_value = float(np.dot(v_ps, v_fh) / norm)
    cos_value = max(-1.0, min(1.0, cos_value))
    aop = math.degrees(math.acos(cos_value))
    return {
        'aop': float(aop),
        'ps_left': tuple(ps_left.tolist()),
        'ps_right': tuple(ps_right.tolist()),
        'fh_contact': tuple(fh_contact.tolist()),
    }


def drawline_AOD(element_, element_1):
    return aop_geometry(element_, element_1)['aop']
