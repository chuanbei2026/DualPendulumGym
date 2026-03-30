import math
import numpy as np
from dual_pendulum_gym.physics.dynamics import PhysicsParams


class PendulumRenderer:
    SCREEN_WIDTH = 1280
    SCREEN_HEIGHT = 900

    def __init__(self, params: PhysicsParams, render_mode: str):
        import pygame
        from pygame.locals import DOUBLEBUF, OPENGL
        from OpenGL.GL import (
            glEnable, glClearColor, glMatrixMode, glLoadIdentity,
            glLight, glLightfv, glColorMaterial, glShadeModel,
            glDepthFunc, glBlendFunc, glLineWidth,
            GL_DEPTH_TEST, GL_LIGHTING, GL_LIGHT0, GL_LIGHT1,
            GL_POSITION, GL_DIFFUSE, GL_AMBIENT, GL_SPECULAR,
            GL_COLOR_MATERIAL, GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE,
            GL_SMOOTH, GL_LESS, GL_BLEND, GL_SRC_ALPHA,
            GL_ONE_MINUS_SRC_ALPHA, GL_PROJECTION, GL_MODELVIEW,
            GL_NORMALIZE,
        )
        from OpenGL.GLU import gluPerspective, gluLookAt

        self.pygame = pygame
        self.params = params
        self.render_mode = render_mode

        pygame.init()
        if render_mode == "human":
            self.screen = pygame.display.set_mode(
                (self.SCREEN_WIDTH, self.SCREEN_HEIGHT), DOUBLEBUF | OPENGL
            )
            pygame.display.set_caption("Dual Pendulum 3D")
        else:
            self.screen = pygame.display.set_mode(
                (self.SCREEN_WIDTH, self.SCREEN_HEIGHT), DOUBLEBUF | OPENGL
            )

        self.clock = pygame.time.Clock()
        self.clock_fps = 50  # default, can be overridden

        # OpenGL setup
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_NORMALIZE)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glClearColor(0.08, 0.08, 0.12, 1.0)

        # Lights
        glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 8.0, 10.0, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.9, 0.9, 0.85, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.25, 0.25, 0.3, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.5, 0.5, 0.5, 1.0])

        glLightfv(GL_LIGHT1, GL_POSITION, [-4.0, 3.0, -5.0, 1.0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.3, 0.3, 0.4, 1.0])
        glLightfv(GL_LIGHT1, GL_AMBIENT, [0.05, 0.05, 0.08, 1.0])

        # Projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(40, self.SCREEN_WIDTH / self.SCREEN_HEIGHT, 0.1, 100.0)

        glMatrixMode(GL_MODELVIEW)

        # Camera position depends on rail length
        self._cam_dist = max(params.rail_limit * 2.0, (params.l1 + params.l2) * 3.0)

        # HUD font (rendered via pygame surface -> texture)
        self._hud_font = pygame.font.SysFont("monospace", 20)

        # Pre-build quadric for cylinders/spheres
        from OpenGL.GLU import gluNewQuadric, gluQuadricNormals, GLU_SMOOTH
        self._quadric = gluNewQuadric()
        gluQuadricNormals(self._quadric, GLU_SMOOTH)

    def _draw_cylinder(self, p1, p2, radius, color):
        """Draw a cylinder from p1 to p2."""
        from OpenGL.GL import glPushMatrix, glPopMatrix, glTranslatef, glRotatef, glColor3f
        from OpenGL.GLU import gluCylinder, gluDisk

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        if length < 1e-8:
            return

        glPushMatrix()
        glColor3f(*color)
        glTranslatef(p1[0], p1[1], p1[2])

        # Rotate from z-axis to direction vector
        ax = -dy
        ay = dx
        az = 0.0
        angle = math.degrees(math.acos(max(-1.0, min(1.0, dz / length))))
        if abs(ax) > 1e-8 or abs(ay) > 1e-8:
            glRotatef(angle, ax, ay, az)
        elif dz < 0:
            glRotatef(180, 1, 0, 0)

        gluCylinder(self._quadric, radius, radius, length, 16, 1)
        # Cap at start
        gluDisk(self._quadric, 0, radius, 16, 1)
        # Cap at end
        glTranslatef(0, 0, length)
        gluDisk(self._quadric, 0, radius, 16, 1)

        glPopMatrix()

    def _draw_sphere(self, center, radius, color):
        from OpenGL.GL import glPushMatrix, glPopMatrix, glTranslatef, glColor3f
        from OpenGL.GLU import gluSphere

        glPushMatrix()
        glColor3f(*color)
        glTranslatef(center[0], center[1], center[2])
        gluSphere(self._quadric, radius, 20, 20)
        glPopMatrix()

    def _draw_box(self, cx, cy, cz, sx, sy, sz, color):
        """Draw an axis-aligned box centered at (cx, cy, cz) with half-sizes (sx, sy, sz)."""
        from OpenGL.GL import (
            glPushMatrix, glPopMatrix, glColor3f,
            glBegin, glEnd, glVertex3f, glNormal3f, GL_QUADS,
        )

        glPushMatrix()
        glColor3f(*color)

        x0, x1 = cx - sx, cx + sx
        y0, y1 = cy - sy, cy + sy
        z0, z1 = cz - sz, cz + sz

        glBegin(GL_QUADS)
        # Front
        glNormal3f(0, 0, 1)
        glVertex3f(x0, y0, z1); glVertex3f(x1, y0, z1)
        glVertex3f(x1, y1, z1); glVertex3f(x0, y1, z1)
        # Back
        glNormal3f(0, 0, -1)
        glVertex3f(x1, y0, z0); glVertex3f(x0, y0, z0)
        glVertex3f(x0, y1, z0); glVertex3f(x1, y1, z0)
        # Top
        glNormal3f(0, 1, 0)
        glVertex3f(x0, y1, z0); glVertex3f(x0, y1, z1)
        glVertex3f(x1, y1, z1); glVertex3f(x1, y1, z0)
        # Bottom
        glNormal3f(0, -1, 0)
        glVertex3f(x0, y0, z0); glVertex3f(x1, y0, z0)
        glVertex3f(x1, y0, z1); glVertex3f(x0, y0, z1)
        # Right
        glNormal3f(1, 0, 0)
        glVertex3f(x1, y0, z0); glVertex3f(x1, y0, z1)
        glVertex3f(x1, y1, z1); glVertex3f(x1, y1, z0)
        # Left
        glNormal3f(-1, 0, 0)
        glVertex3f(x0, y0, z1); glVertex3f(x0, y0, z0)
        glVertex3f(x0, y1, z0); glVertex3f(x0, y1, z1)
        glEnd()

        glPopMatrix()

    def _draw_ground(self):
        """Draw a subtle ground grid."""
        from OpenGL.GL import (
            glPushMatrix, glPopMatrix, glColor4f,
            glBegin, glEnd, glVertex3f, glNormal3f,
            GL_QUADS, glDisable, glEnable, GL_LIGHTING,
            GL_LINES, glLineWidth, glColor3f,
        )

        # Ground plane
        glPushMatrix()
        glColor4f(0.12, 0.12, 0.18, 1.0)
        glBegin(GL_QUADS)
        glNormal3f(0, 1, 0)
        s = 10.0
        y = -self.params.l1 - self.params.l2 - 0.3
        glVertex3f(-s, y, -s); glVertex3f(s, y, -s)
        glVertex3f(s, y, s); glVertex3f(-s, y, s)
        glEnd()
        glPopMatrix()

        # Grid lines
        glDisable(GL_LIGHTING)
        glLineWidth(1.0)
        glColor3f(0.2, 0.2, 0.25)
        glBegin(GL_LINES)
        for i in range(-10, 11):
            glVertex3f(i * 0.5, y + 0.001, -5)
            glVertex3f(i * 0.5, y + 0.001, 5)
            glVertex3f(-5, y + 0.001, i * 0.5)
            glVertex3f(5, y + 0.001, i * 0.5)
        glEnd()
        glEnable(GL_LIGHTING)

    _ACTION_NAMES = {0: "\u25c0 LEFT", 1: "-- NOOP", 2: "RIGHT \u25b6"}

    def _draw_hud(self, state, extra_text=None, info=None):
        """Draw 2D text overlay with force bar on the 3D scene."""
        import pygame
        from OpenGL.GL import (
            glMatrixMode, glPushMatrix, glPopMatrix, glLoadIdentity,
            glDisable, glEnable, glOrtho,
            GL_DEPTH_TEST, GL_LIGHTING, GL_PROJECTION, GL_MODELVIEW,
            GL_TEXTURE_2D, glBindTexture,
            glTexImage2D, glTexParameteri, glGenTextures,
            GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_LINEAR,
            GL_RGBA, GL_UNSIGNED_BYTE,
            glBegin, glEnd, glVertex2f, glTexCoord2f, GL_QUADS,
            glColor4f,
        )

        x, th1, th2, xd, th1d, th2d = state

        # Extract info
        action = info.get("action", 1) if info else 1
        force = info.get("force", 0.0) if info else 0.0
        force_ramp = info.get("force_ramp", 0.0) if info else 0.0
        consec_bal = info.get("consecutive_balanced", 0) if info else 0

        action_name = self._ACTION_NAMES.get(action, "?")
        force_abs = abs(force)

        lines = [
            f"Cart: {x:+.2f}   \u03b81: {np.degrees(th1):+6.1f}\u00b0  \u03b82: {np.degrees(th2):+6.1f}\u00b0",
            f"Action: {action_name}   Force: {force:+.0f}N   Ramp: {force_ramp*100:.0f}%",
            f"Balance streak: {consec_bal}",
        ]
        if extra_text:
            lines.append(extra_text)

        # Render text to a pygame surface
        line_height = 24
        bar_area_h = 30  # space for force bar
        hud_h = line_height * len(lines) + 10 + bar_area_h
        hud_w = 550
        hud_surface = pygame.Surface((hud_w, hud_h), pygame.SRCALPHA)
        hud_surface.fill((0, 0, 0, 140))

        for i, line in enumerate(lines):
            text_surf = self._hud_font.render(line, True, (255, 255, 255))
            hud_surface.blit(text_surf, (10, 5 + i * line_height))

        # Draw force bar at the bottom of the HUD
        bar_y = line_height * len(lines) + 8
        bar_x_center = hud_w // 2
        bar_max_w = 200  # max half-width of bar
        bar_h = 16

        # Background bar outline
        pygame.draw.rect(hud_surface, (60, 60, 70),
                         (bar_x_center - bar_max_w, bar_y, bar_max_w * 2, bar_h), 1)
        # Center tick
        pygame.draw.line(hud_surface, (120, 120, 130),
                         (bar_x_center, bar_y), (bar_x_center, bar_y + bar_h), 2)

        # Force fill bar: left = negative force, right = positive
        if force != 0:
            # Normalize force to [-1, 1] range based on max force (250N)
            f_norm = force / 250.0
            fill_w = int(min(abs(f_norm), 1.0) * bar_max_w)
            # Color: ramp from blue (low) to orange (high)
            ramp = min(force_ramp, 1.0)
            r_col = int(80 + 175 * ramp)
            g_col = int(150 - 80 * ramp)
            b_col = int(220 - 150 * ramp)
            bar_color = (r_col, g_col, b_col)

            if force < 0:  # left
                pygame.draw.rect(hud_surface, bar_color,
                                 (bar_x_center - fill_w, bar_y + 1, fill_w, bar_h - 2))
            else:  # right
                pygame.draw.rect(hud_surface, bar_color,
                                 (bar_x_center, bar_y + 1, fill_w, bar_h - 2))

        # Labels
        left_label = self._hud_font.render("L", True, (180, 180, 190))
        right_label = self._hud_font.render("R", True, (180, 180, 190))
        hud_surface.blit(left_label, (bar_x_center - bar_max_w - 18, bar_y - 3))
        hud_surface.blit(right_label, (bar_x_center + bar_max_w + 6, bar_y - 3))

        # Convert to OpenGL texture
        text_data = pygame.image.tostring(hud_surface, "RGBA", True)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.SCREEN_WIDTH, 0, self.SCREEN_HEIGHT, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_TEXTURE_2D)

        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, hud_w, hud_h, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        glColor4f(1, 1, 1, 1)
        glBegin(GL_QUADS)
        x0, y0 = 10, 10
        glTexCoord2f(0, 0); glVertex2f(x0, y0)
        glTexCoord2f(1, 0); glVertex2f(x0 + hud_w, y0)
        glTexCoord2f(1, 1); glVertex2f(x0 + hud_w, y0 + hud_h)
        glTexCoord2f(0, 1); glVertex2f(x0, y0 + hud_h)
        glEnd()

        glDisable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

        from OpenGL.GL import glDeleteTextures
        glDeleteTextures([tex_id])

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    _STATUS_COLORS = {
        "red": (0.9, 0.15, 0.1),
        "green": (0.1, 0.85, 0.2),
        "blue": (0.15, 0.4, 0.95),
    }

    def render(self, state, extra_text=None, status=None, info=None):
        from OpenGL.GL import glClear, glLoadIdentity, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT
        from OpenGL.GLU import gluLookAt

        x, th1, th2, xd, th1d, th2d = state
        p = self.params

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Camera: slightly above and to the side, following cart loosely
        cam_x = x * 0.3
        gluLookAt(
            cam_x + 1.5, 2.0, self._cam_dist,  # eye
            cam_x, -0.3, 0.0,                    # center
            0.0, 1.0, 0.0,                       # up
        )

        # Ground
        self._draw_ground()

        # Rail: horizontal cylinder along x-axis at y=0
        rail_radius = 0.04
        self._draw_cylinder(
            (-p.rail_limit, 0, 0), (p.rail_limit, 0, 0),
            rail_radius, (0.6, 0.6, 0.65)
        )

        # Rail end posts
        post_h = 0.25
        for rx in [-p.rail_limit, p.rail_limit]:
            self._draw_cylinder(
                (rx, -post_h, 0), (rx, post_h, 0),
                0.05, (0.5, 0.5, 0.55)
            )

        # Rail support legs
        ground_y = -p.l1 - p.l2 - 0.3
        for rx in [-p.rail_limit, p.rail_limit]:
            self._draw_cylinder(
                (rx, 0, 0), (rx, ground_y, 0),
                0.03, (0.35, 0.35, 0.4)
            )

        # Cart: box sitting on the rail
        cart_half_w = 0.25
        cart_half_h = 0.12
        cart_half_d = 0.15
        self._draw_box(x, 0, 0, cart_half_w, cart_half_h, cart_half_d,
                        (0.35, 0.55, 0.8))

        # Wheels (small spheres)
        wheel_y = -cart_half_h - 0.03
        for dx in [-cart_half_w * 0.6, cart_half_w * 0.6]:
            for dz in [-cart_half_d * 0.7, cart_half_d * 0.7]:
                self._draw_sphere((x + dx, wheel_y, dz), 0.04, (0.25, 0.25, 0.3))

        # Rod 1: from cart top center downward
        # Angle th1 from vertical (upright=0), swings in the x-y plane
        rod1_start = (x, 0, 0)
        rod1_end = (
            x + p.l1 * math.sin(th1),
            -p.l1 * math.cos(th1),
            0.0,
        )
        rod_radius = 0.035
        self._draw_cylinder(rod1_start, rod1_end, rod_radius, (0.85, 0.25, 0.2))

        # Rod 2: from rod1 end
        rod2_end = (
            rod1_end[0] + p.l2 * math.sin(th2),
            rod1_end[1] - p.l2 * math.cos(th2),
            0.0,
        )
        self._draw_cylinder(rod1_end, rod2_end, rod_radius * 0.85, (0.2, 0.75, 0.4))

        # Hinge joints (spheres)
        self._draw_sphere(rod1_start, 0.06, (0.9, 0.9, 0.95))
        self._draw_sphere(rod1_end, 0.05, (0.9, 0.9, 0.95))
        self._draw_sphere(rod2_end, 0.035, (0.95, 0.7, 0.2))

        # Status light sphere (floating above the right rail post)
        if status is None:
            from dual_pendulum_gym.envs.dual_pendulum import compute_status
            status = compute_status(state, p)
        light_color = self._STATUS_COLORS.get(status, (0.9, 0.15, 0.1))
        self._draw_sphere((p.rail_limit - 0.3, 1.8, 0), 0.15, light_color)

        # HUD
        self._draw_hud(state, extra_text, info=info)

        if self.render_mode == "human":
            self.pygame.display.flip()
            self.clock.tick(self.clock_fps)
            return None
        else:
            from OpenGL.GL import glReadPixels, GL_RGB, GL_UNSIGNED_BYTE
            data = glReadPixels(0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT,
                                GL_RGB, GL_UNSIGNED_BYTE)
            arr = np.frombuffer(data, dtype=np.uint8).reshape(
                self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3
            )
            return np.flipud(arr)

    def close(self):
        self.pygame.quit()
